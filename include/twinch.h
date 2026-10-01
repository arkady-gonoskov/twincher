// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#pragma once
#include "common.h"
enum twincher_mode{INFERENCE, VARIANCE, QUERY, PURSUIT};

//#define tw_type 0

template<typename fp>
struct twinch_config{
//fp w0, ws, q, z; // permits list of fp variables (assumed by save/load in twincher.h)
    fp wl; // lower bound
    fp ws; // characteristic scale
    fp wu; // upper bound
    // it must be 0 <= wl < ws < wu
    twinch_config(double wl = 0.01, double ws = 0.1, double wu = 1.3):
    wl(wl), ws(ws), wu(wu)
    {
        validate();
    }
    inline void validate() const {
        if(!((0 <= wl)&&(wl < ws)&&(ws < wu)))
            throw std::invalid_argument("twinch_config: it must be 0 <= wl < ws < wu, got wl = " +
                std::to_string(wl) + ", ws = " + std::to_string(ws) + ", wu = " + std::to_string(wu));
    }
    inline void to_vec(std::vector<fp> &vec){
        vec.resize(3); vec[0] = wl; vec[1] = ws; vec[2] = wu;
    }
    inline void from_vec(std::vector<fp> &vec){
        wl = vec[0]; ws = vec[1]; wu = vec[2];
    }
};

template<typename fp>
struct twinch{
    int i0, i1; // indices of affected elements in s vector
    fp c0, c1, w, t; // parameters: location coordinates, width and twist
    fp dc0_da0, dc1_da1, dw_da2, dt_da3; // derivatives set at update
    // fp inv_b; // can be added 
    // fp unused0; // for alignment ?
};

template<typename fp>
HD void twinch_update(twinch<fp> *tw, fp *a, const twinch_config<fp> &tw_config){
    fp param_a0 = 1/sqrt(1 + sqr(a[0]));
    fp param_a1 = 1/sqrt(1 + sqr(a[1]));
    fp param_a3 = 1/sqrt(1 + sqr(a[3]));
    
    tw->c0 = 2*a[0]*param_a0;
    tw->c1 = 2*a[1]*param_a1;
    tw->t = 0.9*a[3]*param_a3;

    tw->dc0_da0 = 2*cube(param_a0);
    tw->dc1_da1 = 2*cube(param_a1);
    tw->dt_da3  = 0.9*cube(param_a3);

    fp wl = tw_config.wl, ws = tw_config.ws, wu = tw_config.wu;
    tw->w = (2*ws + sqr(a[2])*(wu + wl) + a[2]*sqrt(sqr(a[2])*sqr(wu - wl) + 4*(wu - ws)*(ws - wl)))/(2*(1 + sqr(a[2])));
    fp f = (tw->w - wl)*(wu - tw->w);
    tw->dw_da2 = cube(sqrt(f))/(f - 0.5*(tw->w - ws)*(wu + wl - 2*tw->w));
}
template<typename fp>
HD void twinch_alter_config(fp *a, const twinch_config<fp> &tw_config_old, const twinch_config<fp> &tw_config_new){
    // Note that lines commented out concern 
    // parameters that do not alter under the change of w0 and ws, 
    // but this can be changed in further developments.
    
    // fp param_a0 = 1/sqrt(1 + sqr(a[0]));
    // fp param_a1 = 1/sqrt(1 + sqr(a[1]));
    // fp param_a3 = 1/sqrt(1 + sqr(a[3]));
    
    // fp c0 = 2*a[0]*param_a0;
    // fp c1 = 2*a[1]*param_a1;
    // fp t = 0.9*a[3]*param_a3;

    // fp beta0 = c0/2.0;
    // fp beta1 = c1/2.0;
    // fp beta3 = t/0.9;
    
    // a[0] = beta0/sqrt(1 - sqr(beta0));
    // a[1] = beta1/sqrt(1 - sqr(beta1));
    // a[3] = beta3/sqrt(1 - sqr(beta3));

    fp wl = tw_config_old.wl, ws = tw_config_old.ws, wu = tw_config_old.wu;
    fp w = (2*ws + sqr(a[2])*(wu + wl) + a[2]*sqrt(sqr(a[2])*sqr(wu - wl) + 4*(wu - ws)*(ws - wl)))/(2*(1 + sqr(a[2])));
    a[2] = (w - tw_config_new.ws)/sqrt((w - tw_config_new.wl)*(tw_config_new.wu - w));
}


template<twincher_mode Mode, typename fp, int tw_type>
HD inline void twinch_forward(twinch<fp> &tw, fp &s0, fp &s1, fp ds_ds[2][2], fp d2s_dsds[2][2][2]){
    static_assert(tw_type == 0 || tw_type == 1, "unsupported tw_type");
    fp &c0(tw.c0), &c1(tw.c1), &w(tw.w), &t(tw.t);
    [[maybe_unused]] fp &dc0_da0(tw.dc0_da0), &dc1_da1(tw.dc1_da1), &dw_da2(tw.dw_da2), &dt_da3(tw.dt_da3);
//    [[maybe_unused]] fp &inv_b(tw.inv_b);
    
    fp m0 = s0 - c0, m1 = s1 - c1;
    fp b = w*w;
    fp f_b = 1/(b + sqr(m0) + sqr(m1)); 
    fp f = f_b*b;
    fp g;

    if constexpr(tw_type == 0) g = t*f*f;
    else if constexpr(tw_type == 1) g = t*cube(f)*(2 - f*f);

    fp S = 0.5*g*(3 - g*g);
    fp C = sqrt(1 - sqr(S));
    s0 = C*m0 + S*m1 + c0;
    s1 = -S*m0 + C*m1 + c1;
    
    if constexpr (Mode == INFERENCE) return;

    //fp inv_b = 1/b;
    fp inv_C = 1/C;
    fp T = S*inv_C;
    fp dS_dg = 1.5*(1 - g*g);
    fp dg_df;
    if constexpr(tw_type == 0) dg_df = 2*t*f;
    if constexpr(tw_type == 1) dg_df = t*f*f*(6 - 5*f*f);

    fp d2g_df2;
    if constexpr(tw_type == 0) d2g_df2 = 2*t;
    if constexpr(tw_type == 1) d2g_df2 = 4*t*f*(3 - 5*f*f);

    fp dS_df = dS_dg*dg_df;
    fp h = -2*f*f_b;
    fp df_ds0 = h*m0;
    fp df_ds1 = h*m1;
    fp dS_ds0 = dS_df*df_ds0;
    fp dS_ds1 = dS_df*df_ds1;
    fp K = (m1 - T*m0), G = -(T*m1 + m0);
    ds_ds[0][0] = C + K*dS_ds0;
    ds_ds[1][0] = -S + G*dS_ds0;
    ds_ds[0][1] = S + K*dS_ds1;
    ds_ds[1][1] = C + G*dS_ds1;
    if constexpr (Mode == QUERY) return;

    fp C_3 = cube(inv_C);
    fp d2S_df2 = -3*g*sqr(dg_df) + 1.5*(1 - g*g)*d2g_df2;
    fp d2f_ds02 = h*(1 - 4*f_b*sqr(m0));
    fp d2f_ds12 = h*(1 - 4*f_b*sqr(m1));
    fp d2f_ds0ds1 = 8*f*sqr(f_b)*m0*m1;
    fp d2S_ds0ds0 = d2S_df2*sqr(df_ds0) + dS_df*d2f_ds02;
    fp d2S_ds0ds1 = d2S_df2*df_ds1*df_ds0 + dS_df*d2f_ds0ds1;
    fp d2S_ds12 = d2S_df2*sqr(df_ds1) + dS_df*d2f_ds12;

    d2s_dsds[0][0][0] = dS_ds0*(-2*T - m0*dS_ds0*C_3) + K*d2S_ds0ds0;
    d2s_dsds[0][0][1] = dS_ds0 - dS_ds1*(T + m0*C_3*dS_ds0) + K*d2S_ds0ds1;
    d2s_dsds[0][1][0] = d2s_dsds[0][0][1];
    d2s_dsds[0][1][1] = 2*dS_ds1 - C_3*m0*sqr(dS_ds1) + K*d2S_ds12;
    d2s_dsds[1][0][0] = -dS_ds0*(2 + m1*C_3*dS_ds0) + G*d2S_ds0ds0;
    d2s_dsds[1][0][1] = -dS_ds1 - dS_ds0*(T + m1*C_3*dS_ds1) + G*d2S_ds0ds1;
    d2s_dsds[1][1][0] = d2s_dsds[1][0][1];
    d2s_dsds[1][1][1] = -dS_ds1*(2*T + m1*C_3*dS_ds1) + G*d2S_ds12;    
}

template<twincher_mode Mode, typename fp, int tw_type>
HD inline void twinch_backward(twinch<fp> &tw, fp &s0, fp &s1, fp ds_ds[2][2], fp d2s_dsds[2][2][2], fp ds_da[2][4], fp d2s_dsda[2][2][4]){
    static_assert(tw_type == 0 || tw_type == 1, "unsupported tw_type");
    fp &c0(tw.c0), &c1(tw.c1), &w(tw.w), &t(tw.t);
    [[maybe_unused]] fp &dc0_da0(tw.dc0_da0), &dc1_da1(tw.dc1_da1), &dw_da2(tw.dw_da2), &dt_da3(tw.dt_da3);
//    [[maybe_unused]] fp &inv_b(tw.inv_b);

    fp mr0 = s0 - c0, mr1 = s1 - c1;
    fp b = w*w;
    fp m2 = sqr(mr0) + sqr(mr1);
    fp f_b = 1/(b + m2); 
    fp f = f_b*b;
    
    fp g;
    if constexpr(tw_type == 0) g = t*f*f;
    else if constexpr(tw_type == 1) g = t*cube(f)*(2 - f*f);

    fp S = -0.5*g*(3 - g*g); // reverse rotation
    fp C = sqrt(1 - sqr(S));
    s0 = C*mr0 + S*mr1 + c0;
    s1 = -S*mr0 + C*mr1 + c1;

    if constexpr (Mode == INFERENCE) return;

    fp m0 = s0 - c0, m1 = s1 - c1;
    S = -S; // changing back to forward rotation
    //fp inv_b = 1/b;
    fp inv_C = 1/C;
    fp T = S*inv_C;
    fp dS_dg = 1.5*(1 - g*g);
    
    fp dg_df;
    if constexpr(tw_type == 0) dg_df = 2*t*f;
    if constexpr(tw_type == 1) dg_df = t*f*f*(6 - 5*f*f);

    fp d2g_df2;
    if constexpr(tw_type == 0) d2g_df2 = 2*t;
    if constexpr(tw_type == 1) d2g_df2 = 4*t*f*(3 - 5*f*f);

    fp dS_df = dS_dg*dg_df;
    fp h = -2*f*f_b;
    fp df_ds0 = h*m0;
    fp df_ds1 = h*m1;
    fp dS_ds0 = dS_df*df_ds0;
    fp dS_ds1 = dS_df*df_ds1;
    fp K = (m1 - T*m0), G = -(T*m1 + m0);

    ds_ds[0][0] = C + K*dS_ds0;
    ds_ds[1][0] = -S + G*dS_ds0;
    ds_ds[0][1] = S + K*dS_ds1;
    ds_ds[1][1] = C + G*dS_ds1;

    if constexpr (Mode == VARIANCE) return;
// upt to here 44 registers
    fp C_3 = cube(inv_C);
    

    fp d2S_df2 = -3*g*sqr(dg_df) + 1.5*(1 - g*g)*d2g_df2;

    fp d2f_ds02 = h*(1 - 4*f_b*sqr(m0));
    fp d2f_ds12 = h*(1 - 4*f_b*sqr(m1));
    fp d2f_ds0ds1 = 8*f*sqr(f_b)*m0*m1;
    fp d2S_ds0ds0 = d2S_df2*sqr(df_ds0) + dS_df*d2f_ds02;
    fp d2S_ds0ds1 = d2S_df2*df_ds1*df_ds0 + dS_df*d2f_ds0ds1;
    fp d2S_ds12 = d2S_df2*sqr(df_ds1) + dS_df*d2f_ds12;

    d2s_dsds[0][0][0] = dS_ds0*(-2*T - m0*dS_ds0*C_3) + K*d2S_ds0ds0;
    d2s_dsds[0][0][1] = dS_ds0 - dS_ds1*(T + m0*C_3*dS_ds0) + K*d2S_ds0ds1;
    d2s_dsds[0][1][0] = d2s_dsds[0][0][1];
    d2s_dsds[0][1][1] = 2*dS_ds1 - C_3*m0*sqr(dS_ds1) + K*d2S_ds12;
    d2s_dsds[1][0][0] = -dS_ds0*(2 + m1*C_3*dS_ds0) + G*d2S_ds0ds0;
    d2s_dsds[1][0][1] = -dS_ds1 - dS_ds0*(T + m1*C_3*dS_ds1) + G*d2S_ds0ds1;
    d2s_dsds[1][1][0] = d2s_dsds[1][0][1];
    d2s_dsds[1][1][1] = -dS_ds1*(2*T + m1*C_3*dS_ds1) + G*d2S_ds12;    

    fp df_dc0 = 2*f_b*f*m0;
    fp df_dc1 = 2*f_b*f*m1;
    fp dS_dc0 = dS_df*df_dc0;
    fp dS_dc1 = dS_df*df_dc1;
    fp df_dw = 2*w*sqr(f_b)*m2;
    fp dS_dw = dS_df*df_dw;

    fp dg_dt;
    if constexpr(tw_type == 0) dg_dt = f*f;
    if constexpr(tw_type == 1) dg_dt = cube(f)*(2 - f*f);
    fp dS_dt = dS_dg*dg_dt;

    ds_da[0][0] = dc0_da0*(1 - C + K*dS_dc0);
    ds_da[0][1] = dc1_da1*(-S + K*dS_dc1);
    ds_da[0][2] = dw_da2*(K*dS_dw);
    ds_da[0][3] = dt_da3*(K*dS_dt);
    ds_da[1][0] = dc0_da0*(S + G*dS_dc0);
    ds_da[1][1] = dc1_da1*(1 - C + G*dS_dc1);
    ds_da[1][2] = dw_da2*(G*dS_dw);
    ds_da[1][3] = dt_da3*(G*dS_dt);

    fp dK_dc0 = T - m0*C_3*dS_dc0;
    fp dK_dc1 = -1 - m0*C_3*dS_dc1;
    fp dG_dc0 = 1 - m1*C_3*dS_dc0;
    fp dG_dc1 = T - m1*C_3*dS_dc1;
    fp dK_dw = -m0*C_3*dS_dw;
    fp dG_dw = -m1*C_3*dS_dw;
    fp dK_dt = -m0*C_3*dS_dt;
    fp dG_dt = -m1*C_3*dS_dt;

    fp coef1 = -3*g*sqr(dg_df) + dS_dg*d2g_df2;
    fp d2S_ds0dc0 = coef1*df_ds0*df_dc0 + dS_dg*dg_df*(-d2f_ds02);
    fp d2S_ds1dc1 = coef1*df_ds1*df_dc1 + dS_dg*dg_df*(-d2f_ds12);
    fp d2S_ds0dc1 = coef1*df_ds0*df_dc1 + dS_dg*dg_df*(-d2f_ds0ds1);
    fp d2S_ds1dc0 = coef1*df_ds1*df_dc0 + dS_dg*dg_df*(-d2f_ds0ds1);

    fp d2f_ds0dw_m0 = sqr(sqr(f_b))*4*w*(sqr(b) - sqr(m2));   //(b - sqr(m0) - sqr(m1))*(b + sqr(m0) + sqr(m1));
    fp d2S_ds0dw = coef1*df_ds0*df_dw + dS_dg*dg_df*d2f_ds0dw_m0*m0;
    fp d2S_ds1dw = coef1*df_ds1*df_dw + dS_dg*dg_df*d2f_ds0dw_m0*m1;

    fp d2g_dfdt;
    if constexpr(tw_type == 0) d2g_dfdt = 2*f;
    if constexpr(tw_type == 1) d2g_dfdt = f*f*(6 - 5*f*f);
    
    fp d2S_ds0dt = (-3*g*dg_dt*dg_df + dS_dg*d2g_dfdt)*df_ds0;
    fp d2S_ds1dt = (-3*g*dg_dt*dg_df + dS_dg*d2g_dfdt)*df_ds1;     

    d2s_dsda[0][0][0] = dc0_da0*(-T*dS_dc0 + K*d2S_ds0dc0 + dS_ds0*dK_dc0);
    d2s_dsda[0][0][1] = dc1_da1*(-T*dS_dc1 + K*d2S_ds0dc1 + dS_ds0*dK_dc1);
    d2s_dsda[0][0][2] = dw_da2*(-T*dS_dw + K*d2S_ds0dw + dS_ds0*dK_dw);
    d2s_dsda[0][0][3] = dt_da3*(-T*dS_dt + K*d2S_ds0dt + dS_ds0*dK_dt);

    d2s_dsda[0][1][0] = dc0_da0*(dS_dc0 + K*d2S_ds1dc0 + dS_ds1*dK_dc0);
    d2s_dsda[0][1][1] = dc1_da1*(dS_dc1 + K*d2S_ds1dc1 + dS_ds1*dK_dc1);
    d2s_dsda[0][1][2] = dw_da2*(dS_dw + K*d2S_ds1dw + dS_ds1*dK_dw); 
    d2s_dsda[0][1][3] = dt_da3*(dS_dt + K*d2S_ds1dt + dS_ds1*dK_dt);

    d2s_dsda[1][0][0] = dc0_da0*(-dS_dc0 + G*d2S_ds0dc0 + dS_ds0*dG_dc0); 
    d2s_dsda[1][0][1] = dc1_da1*(-dS_dc1 + G*d2S_ds0dc1 + dS_ds0*dG_dc1); 
    d2s_dsda[1][0][2] = dw_da2*(-dS_dw + G*d2S_ds0dw + dS_ds0*dG_dw); 
    d2s_dsda[1][0][3] = dt_da3*(-dS_dt + G*d2S_ds0dt + dS_ds0*dG_dt); 

    d2s_dsda[1][1][0] = dc0_da0*(-T*dS_dc0 + G*d2S_ds1dc0 + dS_ds1*dG_dc0);
    d2s_dsda[1][1][1] = dc1_da1*(-T*dS_dc1 + G*d2S_ds1dc1 + dS_ds1*dG_dc1);
    d2s_dsda[1][1][2] = dw_da2*(-T*dS_dw + G*d2S_ds1dw + dS_ds1*dG_dw);
    d2s_dsda[1][1][3] = dt_da3*(-T*dS_dt + G*d2S_ds1dt + dS_ds1*dG_dt);
}
