# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

def register_components(register):
    from .hs_loss import HSLoss
    from .hs_noise_loss import HSNoiseLoss

    register("loss_terms", "hs", HSLoss)
    register("loss_terms", "hs_noise", HSNoiseLoss)

    register("arch_defaults", "hs", {
        "hs" : {
            "det_margin" : 0.8, # target margin for determinants
            "amp" : 3.0, # amplitude of loss
            "det_clearance" : 0.5, # threshold for triggering expansion
            "p_range_step" : 0.01, # increment for p_range
            "ramp_boost" : 2.0, # loss amplitude boost at the periphery 
            "ramp_size" : 0.2, # size of the ramp for the loss boost
            "trimmer_threshold" : 1.3, # threshold for the loss on velocities
            "trimmer_amp" : 1.0, # amplitude of loss for velocities 
            "promotion_threshold" : 10, # the number of valid-state iterations to pass until promotion
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
