# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

from pathlib import Path
import numpy as np
from . import tw_util
from .shuttle import Shuttle
from .interpolator import TorchMultilinearInterpolator
import json, torch
from . import registry

def _get_loss_term(loss_name, **kwargs):
    if loss_name not in registry.components("loss_terms"):
        raise ValueError(f"unknown loss name {loss_name}")
    else:
        return registry.components("loss_terms")[loss_name](**kwargs)

def _modify_params(obj, params: dict):
    unknown = [key for key in params if not hasattr(obj, key)]
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")

    for key, value in params.items():
        setattr(obj, key, value)

class Learner:
    def __init__(self, 
        arch: str, # architecture ("hs"/"as"/"gs") 
        n_s: int, # dimensionality of the twincher state-vector
        n_l: int, # number of twincher layers
        lr: float = 1e-3, # base learning rate
        n_c: int = 5*1024, # number of cases in the training batch
        device: str = "auto", # the alternative is "cpu"
        dir = None, # directory for data output (str or Path; default: ./twincher_output)
        output_stride: int = 200, # stride for updating states and learning curve
        adam_params: dict = {"betas": (0.99, 0.999)}, # adam optimizer parameters
        damper: dict = {"cosine_min": -0.5, "down_factor": 0.5, "up_factor": 1.1, "lr_min": 1e-5}, # damper
        time_monitor = "auto", # set to None for no time monitoring (can be faster) 
        plot_progress = True, # True enables updates of the learning curve and other outputs every state_stride 
        rng_seed = 42, # rng seed for sampling batches
    ):
        self.arch = arch
        self.n_s = n_s
        self.n_l = n_l
        self.lr = lr
        self.n_c = n_c
        device = tw_util.select_device(device)
        self.device = device
        self.dtype = torch.float32 if device == "cuda" else torch.float64 
        self.dir = dir
        self.output_stride = output_stride
        self.adam_params = adam_params
        self.damper = damper
        self.time_monitor = time_monitor
        self.plot_progress = plot_progress
        self.rng_seed = rng_seed

        self.data_collected = False
        self.initiated = False
        self.loss_terms = []

        if arch not in registry.components("arch_defaults"):
            raise ValueError(f"Unsupported arch '{arch}'")
        else:
            self.modify_params({"loss_terms" : registry.components("arch_defaults")[arch]})

    def modify_params(self, params: dict):
        if self.initiated:
            raise RuntimeError("modify_params can only be called before init() and the first step()")

        unknown = [key for key in params if not hasattr(self, key)]
        if unknown:
            raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")

        for key, value in params.items():
            if key == "adam_params":
                self.adam_params = {**self.adam_params, **params[key]}
            elif key == "damper":
                self.damper = {**self.damper, **params[key]}
            elif key == "loss_terms":
                for term_name, term_params in params[key].items():
                    term_ind = None
                    for i in range(len(self.loss_terms)):
                        if self.loss_terms[i].name == term_name:
                            term_ind = i
                    if (term_ind is None) and (term_params is not None):
                        self.loss_terms.append(_get_loss_term(term_name, **term_params))
                    elif (term_ind is not None) and (term_params is None):
                        del self.loss_terms[term_ind]
                    elif (term_ind is not None) and (term_params is not None):
                        _modify_params(self.loss_terms[term_ind], term_params)
            else:
                setattr(self, key, value)


    def get_p_request(self, n_p, n_grid):
        self.n_grid = n_grid
        from itertools import product
        return np.array(list(product(np.linspace(-1, 1, n_grid), repeat=n_p)), dtype=np.float64)

    def set_data(self, y, dy_dp, stencil_name = "none"):
        self.n_data = dy_dp.shape[0]
        self.n_y = dy_dp.shape[1]
        self.n_p = dy_dp.shape[2]
        if y.shape != (self.n_data, self.n_y):
            raise ValueError("inconsistent shape of y and dy_dp")
        self.data_y = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        self.data_dy_dp = torch.as_tensor(dy_dp, dtype=self.dtype, device=self.device)
        self.stencil_name = stencil_name
        self.data_collected = True

    def generate_data(self, stencil, n_grid, dp_eval = 1e-8):
        self.n_grid = n_grid
        self.dp_eval = dp_eval
        self.n_p = stencil.n_p
        self.n_y = stencil.n_y
        n_p, n_y = self.n_p, self.n_y

        from itertools import product
        h_data_p = np.array(list(product(np.linspace(-1, 1, n_grid), repeat=n_p)), dtype=np.float64)
        self.n_data = h_data_p.shape[0]
        n_data = self.n_data
        
        h_data_y = np.zeros((n_data, n_y), dtype=np.float64)
        h_data_dy_dp = np.zeros((n_data, n_y, n_p), dtype=np.float64)
        for i_d in range(n_data):
            tw_util.estimate_J(
                stencil=stencil,
                p=h_data_p[i_d, :],
                y=h_data_y[i_d, :],
                dy_dp=h_data_dy_dp[i_d, :, :],
                d_eval=self.dp_eval,
            )
        self.data_y = torch.as_tensor(h_data_y, dtype=self.dtype, device=self.device)
        self.data_dy_dp = torch.as_tensor(h_data_dy_dp, dtype=self.dtype, device=self.device)
        self.stencil_name = stencil.name
        self.data_collected = True

    def set_stencil(self, stencil):
        self.n_p = stencil.n_p
        self.n_y = stencil.n_y
        self.stencil_name = stencil.name
        if not hasattr(stencil, "compute"):
            raise RuntimeError("stencil must provide compute(p, y, dy_dp)")
        self.stencil = stencil
        self.data_collected = True

    def init(self):
        """
        Prepare the training: create the twincher (self.SH), the loss terms and the optimizer.

        This is done automatically by the first call of step(). Calling init() explicitly
        gives access to the twincher before the first step, for example to set up a monitor.
        Afterwards, the parameters can no longer be changed with modify_params().
        """
        if self.initiated:
            raise RuntimeError("init() has already been called")
        if not self.data_collected:
            raise RuntimeError(
                "missing data (use get_p_request() + set_data(), generate_data(stencil) or set_stencil(stencil))"
            )

        self.damper, _ = tw_util.validate_parameters(self.damper, {
            "cosine_min": (-1.0, 0.0, -0.5), 
            "down_factor": (0.0, 1.0, 0.5), 
            "up_factor": (1.0, 2.0, 1.1), 
            "lr_min": (0, None, 1e-5), 
        }, raise_err=True)

        self.dir = tw_util.output_dir(self.dir)

        # configuration as JSON (__str__ returns it after initialization, when the full
        # state of the learner is no longer serializable)
        self._config_json = self._to_json()
        with open(self.dir / "learner.json", "w", encoding="utf-8") as f:
            f.write(self._config_json)

        # set time_monitor; without time monitoring, a disabled monitor is used, so that
        # components (e.g. loss terms) can call its start() and stop() unconditionally
        from .time_monitor import TimeMonitor
        if self.time_monitor == "auto":
            self.time_monitor = TimeMonitor(str(self.dir) + "/learner_tm.log")
        elif self.time_monitor is None:
            self.time_monitor = TimeMonitor(file_name=None, disable=True)
        self.time_monitor.start("learner::init")

        # set internal variables
        self.iter = 0
        self.lr_dynamic = self.lr
        self.n_m = max(
            (loss.n_m_req for loss in self.loss_terms if hasattr(loss, "n_m_req")),
            default=self.n_p,
        )
        n_s, n_l, n_c, n_p, n_m, n_y = self.n_s, self.n_l, self.n_c, self.n_p, self.n_m, self.n_y    

        self.p = torch.empty((n_c, n_p), dtype=self.dtype, device=self.device)
        self.y = torch.empty((n_c, n_y), dtype=self.dtype, device=self.device)
        self.dy_dp = torch.empty((n_c, n_y, n_p), dtype=self.dtype, device=self.device)

        if not hasattr(self, "stencil"):
            self.MLI = TorchMultilinearInterpolator(n_p=n_p, n_g=self.n_grid, device=self.device)

        # random number generator
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.rng_seed)

        # initial range for batch generation 
        self.p_range = 1.0

        # initiate loss terms:
        for term in self.loss_terms:
            term.init(self)        

        # initiate shuttle
        self.SH = Shuttle(
            data_type= "torch" if self.device == "cuda" else "torch_cpu", 
            n_s=n_s, n_l=n_l, n_c=n_c, n_p=n_p, n_m=n_m,
        )
        if not getattr(self.time_monitor, "disable", False):
            self.SH.enable_time_monitor(time_monitor=self.time_monitor)
        self.n_t = self.SH.n_t # number of twinches

        # main gradient collector and total loss
        self.g = torch.empty((self.n_t*4,), dtype=self.dtype, device=self.device)
        self.loss = None

        # initiate optimizer
        self.optimizer = torch.optim.Adam(
            [self.SH.param],
            lr=self.lr,
            **self.adam_params,
        )

        if self.plot_progress:
            from .learning_curve import LearningCurve
            self.LC = LearningCurve(stride=self.output_stride)

        self.initiated = True
        self.time_monitor.stop("learner::init")

    # former name of init(), kept for compatibility
    _init = init

    def _damper(self, make_output: bool):
        if not hasattr(self, "prev_g"):
            self.prev_g = torch.zeros((self.n_t*4,), dtype=self.dtype, device=self.device)
            self.prev_g_norm = 1.0            
            return
        self.time_monitor.start("learner::damper", torch_sync=True)
        norm_g = torch.linalg.norm(self.g)
        c = torch.dot(self.prev_g, self.g)/(self.prev_g_norm * norm_g + 1e-12)
        if hasattr(self, "c_prev") == False:
            self.c_prev = 0.0
        if hasattr(self, "damper_count") == False:
            self.damper_count = 0
            self.damper_lr_cum = 0
            self.damper_n_down = 0
        if c < self.damper["cosine_min"] and self.c_prev < self.damper["cosine_min"]:
            self.damper_n_down += 1
            self.lr_dynamic = max(self.damper["lr_min"], self.damper["down_factor"]*self.lr_dynamic) # fast down
        elif c > 0.0: 
            self.lr_dynamic = min(self.lr, self.damper["up_factor"]*self.lr_dynamic) # slow up            
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr_dynamic
        self.c_prev = c
        self.prev_g.copy_(self.g)
        self.prev_g_norm = norm_g
        self.damper_lr_cum += self.lr_dynamic
        self.damper_count += 1
        if make_output:
            self.state += f"damper[{self.damper_n_down}, {self.damper_lr_cum/self.damper_count:7.2e}]"
            self.damper_count = 0
            self.damper_lr_cum = 0
            self.damper_n_down = 0
        self.time_monitor.stop("learner::damper", torch_sync=True)

    def step(self, test = False):

        if not self.initiated:
            self.init()

        self.time_monitor.start("learner::step", torch_sync=True)

        make_output = self.iter % self.output_stride == 0
        if make_output:
            self.state = ""

        # generate batch
        self.p.uniform_(-self.p_range, self.p_range, generator=self.rng)
        if not hasattr(self, "stencil"):
            self.MLI.interpolate(self.p, self.y, self.data_y)
            self.MLI.interpolate(self.p, self.dy_dp[:, :, :], self.data_dy_dp)
        else:
            self.stencil.compute(self.p, self.y, self.dy_dp)

        # computing cumulative gradient
        self.g.fill_(0)
        for term in self.loss_terms:
            # computer gradient using self.SH and place gradient in self.SH.g:
            term.compute_grad(make_output, test=test)

            self.g[:] += self.SH.g[:]

        if self.damper is not None:
            self._damper(make_output)

        # run optimization step
        if test:
            self.SH.a[:] -= self.lr*self.g[:]
        else:
            self.SH.g.copy_(self.g)
            self.optimizer.step()
        self.SH.tw_update() # updates internal state

        self.loss_detailed = ""
        self.loss = 0
        for term in self.loss_terms:
            self.loss += term.loss
            self.loss_detailed += f"{term.loss:10.8f}|"
        self.loss_detailed += f"{self.loss:10.8f}"

        if make_output:
            self.state += " "
            for term in self.loss_terms:        
                self.state += (term.name + ":" + term.state + " ")

        if self.plot_progress:
            self.LC(self)

        self.iter += 1

        self.time_monitor.stop("learner::step", torch_sync=True)

    _json_exclude = ["data_y", "data_dy_dp", "stencil", "_config_json"]
    def __str__(self):
        if self.initiated:
            return self._config_json
        return self._to_json()

    def _to_json(self):
        data = {
            key: value
            for key, value in self.__dict__.items()
            if key not in self._json_exclude
        }

        string_types = (Path, torch.dtype)

        def json_default(obj):
            if isinstance(obj, string_types):
                return str(obj)
            return obj.__dict__

        return json.dumps(
            data,
            indent=4,
            default=json_default,
        )

if __name__ == "__main__":
    L = Learner(arch="hs", n_s=64, n_l=64)
    #print(L)