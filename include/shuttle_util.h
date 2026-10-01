// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#pragma once
#include "twinch.h"

template<twincher_mode Mode, typename fp, int tw_type>
HD void twincher_forward(twinch<fp> &tw_, fp *s, fp *_Q, int ic, int n_c, int n_s, int n_m){
    twinch tw = tw_; // upload all at once
    fp ds_ds[2][2];
    fp d2s_dsds[2][2][2];
    twinch_forward<Mode, fp, tw_type>(tw, s[tw.i0*n_c + ic], s[tw.i1*n_c + ic], ds_ds, d2s_dsds);
    if constexpr (Mode == INFERENCE) return;
    tensor3_view Q(_Q, n_s, n_m, n_c);
    int i0 = tw.i0, i1 = tw.i1;
    //updating Q (ds_dp): Mode == Query
    for(int im = 0; im < n_m; im++){
        fp tmp = ds_ds[0][0]*Q(i0, im, ic) + ds_ds[0][1]*Q(i1, im, ic);
        Q(i1, im, ic) = ds_ds[1][0]*Q(i0, im, ic) + ds_ds[1][1]*Q(i1, im, ic);
        Q(i0, im, ic) = tmp;
    }
};

template<twincher_mode Mode, typename fp, int tw_type>
HD void twincher_backward(twinch<fp> &tw_, fp *s, fp *_Q, fp *_B, fp *_A, fp *_dL_ds, fp *_dLc_da, fp sga, fp sgt, fp inv_n_c, int ic, int n_c, int n_s, int n_m){
    twinch tw = tw_; // upload all at once
    
    fp ds_ds[2][2];
    fp d2s_dsds[2][2][2];
    fp ds_da[2][4];
    fp d2s_dsda[2][2][4];

    int i0 = tw.i0, i1 = tw.i1;
    tensor2_view dL_ds(_dL_ds, n_s, n_c);
    
    //accounting for the s-gate regularization loss
    if constexpr (Mode != INFERENCE) {
        fp s0 = s[tw.i0*n_c + ic], s1 = s[tw.i1*n_c + ic];
        fp s0_abs = std::abs(s0), s1_abs = std::abs(s1);
        dL_ds(i0, ic) += 2*sga*(s0_abs - sgt)*sgn(s0)*(s0_abs > sgt);
        dL_ds(i1, ic) += 2*sga*(s1_abs - sgt)*sgn(s1)*(s1_abs > sgt);                
    }

    twinch_backward<Mode, fp, tw_type>(tw, s[tw.i0*n_c + ic], s[tw.i1*n_c + ic], ds_ds, d2s_dsds, ds_da, d2s_dsda);
    if constexpr (Mode == INFERENCE) return;
    tensor3_view Q(_Q, n_s, n_m, n_c);
    tensor3_view B(_B, n_s, n_m, n_c);
    tensor2_view A(_A, n_s, n_c);
    tensor2_view dLc_da(_dLc_da, 4, n_c);

    //updating Q:
    fp inv_ds_ds[2][2];
    inv22<fp>(ds_ds, inv_ds_ds);
    for(int im = 0; im < n_m; im++){
        fp tmp = inv_ds_ds[0][0]*Q(i0, im, ic) + inv_ds_ds[0][1]*Q(i1, im, ic);
        Q(i1, im, ic) = inv_ds_ds[1][0]*Q(i0, im, ic) + inv_ds_ds[1][1]*Q(i1, im, ic);
        Q(i0, im, ic) = tmp;
    }
    
    //computing dLc_da (g contributions for individual ic):
    {
        for(int ia = 0; ia < 4; ia++) 
            dLc_da(ia, ic) = A(i0, ic)*ds_da[0][ia] + A(i1, ic)*ds_da[1][ia];

        for(int im = 0; im < n_m; im++)
        for(int ia = 0; ia < 4; ia++)
        {
            dLc_da(ia, ic) += B(i0, im, ic)*Q(i0, im, ic)*d2s_dsda[0][0][ia];
            dLc_da(ia, ic) += B(i1, im, ic)*Q(i0, im, ic)*d2s_dsda[1][0][ia];
            dLc_da(ia, ic) += B(i0, im, ic)*Q(i1, im, ic)*d2s_dsda[0][1][ia];
            dLc_da(ia, ic) += B(i1, im, ic)*Q(i1, im, ic)*d2s_dsda[1][1][ia];
        }

        //accounting for dLc_ds:
        for(int ia = 0; ia < 4; ia++)
            dLc_da(ia, ic) += dL_ds(i0, ic)*ds_da[0][ia] + dL_ds(i1, ic)*ds_da[1][ia];

        //normalizing dLc_da:
        for(int ia = 0; ia < 4; ia++)
            dLc_da(ia, ic) *= inv_n_c;
    }
    
    //updating A:
    {
        fp Ai0 = A(i0, ic)*ds_ds[0][0] + A(i1, ic)*ds_ds[1][0];
        A(i1, ic) = A(i0, ic)*ds_ds[0][1] + A(i1, ic)*ds_ds[1][1];
        A(i0, ic) = Ai0;

        for(int im = 0; im < n_m; im++){
            A(i0, ic) += B(i0, im, ic)*d2s_dsds[0][0][0]*Q(i0, im, ic);
            A(i0, ic) += B(i1, im, ic)*d2s_dsds[1][0][0]*Q(i0, im, ic);
            A(i0, ic) += B(i0, im, ic)*d2s_dsds[0][0][1]*Q(i1, im, ic);
            A(i0, ic) += B(i1, im, ic)*d2s_dsds[1][0][1]*Q(i1, im, ic);
            A(i1, ic) += B(i0, im, ic)*d2s_dsds[0][1][0]*Q(i0, im, ic);
            A(i1, ic) += B(i1, im, ic)*d2s_dsds[1][1][0]*Q(i0, im, ic);
            A(i1, ic) += B(i0, im, ic)*d2s_dsds[0][1][1]*Q(i1, im, ic);
            A(i1, ic) += B(i1, im, ic)*d2s_dsds[1][1][1]*Q(i1, im, ic);
        }
    }

    //updating B:
    for(int im = 0; im < n_m; im++){
        fp tmp = ds_ds[0][0]*B(i0, im, ic) + ds_ds[1][0]*B(i1, im, ic);
        B(i1, im, ic) = ds_ds[0][1]*B(i0, im, ic) + ds_ds[1][1]*B(i1, im, ic);
        B(i0, im, ic) = tmp;
    }

    //updating dL_ds:
    {
        fp dL_ds_i0 = dL_ds(i0, ic)*ds_ds[0][0] + dL_ds(i1, ic)*ds_ds[1][0];
        dL_ds(i1, ic) = dL_ds(i0, ic)*ds_ds[0][1] + dL_ds(i1, ic)*ds_ds[1][1];
        dL_ds(i0, ic) = dL_ds_i0; 
    }
}

template<typename fp, int tw_type>
HD void twincher_backward_V(twinch<fp> &tw_, fp *s, fp *_V, int ic, int n_p, int n_s, int n_c){
    twinch tw = tw_; // upload all at once
    fp ds_ds[2][2];
    twinch_backward<VARIANCE, fp, tw_type>(tw, s[tw.i0*n_c + ic], s[tw.i1*n_c + ic], ds_ds, nullptr, nullptr, nullptr);    
    int i0 = tw.i0, i1 = tw.i1;
    tensor3_view V(_V, n_p, n_s, n_c);
    for(int ip = 0; ip < n_p; ip++){
        fp tmp = V(ip, i0, ic)*ds_ds[0][0] + V(ip, i1, ic)*ds_ds[1][0];
        V(ip, i1, ic) = V(ip, i0, ic)*ds_ds[0][1] + V(ip, i1, ic)*ds_ds[1][1];
        V(ip, i0, ic) = tmp;
    }
}