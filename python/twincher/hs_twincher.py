# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

def register_components(register):
    from .hs_loss import HSLoss
    from .hs_noise_loss import HSNoiseLoss
    from .hs_sm_noise_loss import HSSMNoiseLoss

    register("loss_terms", "hs", HSLoss)
    register("loss_terms", "hs_noise", HSNoiseLoss)
    register("loss_terms", "hs_sm_noise", HSSMNoiseLoss)

    register("arch_defaults", "hs", {
        "hs" : {
            "det_margin" : 0.8, # target margin for determinants
            "amp" : 3.0, # amplitude of loss
            "det_clearance" : 0.5, # threshold for triggering expansion
            "p_range_step" : 0.01, # increment for p_range
            "ramp_boost" : 1.0, # loss amplitude factor at the border of the box
            "ramp_size" : 0.2, # width of the ramp of the loss amplitude (and initial p_range)
            "trimmer_threshold" : 1.3, # threshold for the loss on velocities
            "trimmer_amp" : 1.0, # amplitude of loss for velocities
            "promotion_threshold" : 10, # the number of valid-state iterations to pass until promotion
            "trimmer_domain_amp" : 1.0, # amplitude of the trimmer over the whole domain (0: off)
            "momentum_reset" : None, # reset of Adam's momentum at loss spikes (True or settings)
        },
        "hs_noise" : {
            "amp": 1e-4,
            "vel_scale": 1.0,
            "y_range": 0.5,     
        }
    })

    from .hs_solver import HSSolver
    register("solvers", "hs", HSSolver)

    from .hs_monitor_p2 import HSMonitorP2
    register("monitors", "hs_monitor_p2", HSMonitorP2)
