 /*
  This file is part of the ymir Library.
  ymir is a C library for modeling ice sheets

  Copyright (C) 2010, 2011 Carsten Burstedde, Toby Isaac, Johann Rudi,
                           Georg Stadler, Lucas Wilcox.

  The ymir Library is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Founisdation, either version 3 of the License, or
  (at your option) any later version.

  The ymir Library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the ymir Library.  If not, see <http://www.gnu.org/licenses/>.

  ---

  This is the slabs example for global instantaneous mantle flow with plates.

*/

#include <slabs_base.h>
#include <slabs_norm.h>
#include <slabs_physics_extended.h>
#include <slabs_discretization.h>
#include <slabs_discretization_extended.h>
#include <slabs_linear_stokes_problem.h>
#include <slabs_nonlinear_stokes_problem.h>
#include <slabs_stokes_state.h>
#include <slabs_io.h>
#include <slabs_vtk.h>
#include <slabs_gradient.h>
#include <slabs_hessian.h>
#include <slabs_stokes_state.h>
#include <ymir_gmg_hierarchy_mesh.h>
#include <ymir_gmg_hierarchy_stress.h>
#include <ymir_gmg_hierarchy_bbt.h>
#include <ymir_perf_counter.h>

// for nonlinear solver
#include <ymir_velocity_vec.h>

//###DEV###
#include <ymir_interp_vec.h>
#include <ymir_mass_vec.h>
#include <ymir_mass_elem.h>
#include <ymir_nl.h>
#include <ymir_pressure_vec.h>
#include <ymir_derivative_elem.h>
#include <ymir_velocity_vec.h>
#include <ymir_interp_vec.h>
#include <ymir_vec_getset.h>
#include <ymir_vtk.h>



/* This is the W block in the Hessian */
#if 1
void
slabs_hessian_block_uu (slabs_inverse_problem_params_t *inverse_params,
			slabs_hessian_params_t *hessian_params)

{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc;
  ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem;
  ymir_cvec_t   *intrace = ymir_face_cvec_new (mesh, SL_TOP, 3);
  ymir_cvec_t   *intrace_mass = ymir_face_cvec_new (mesh, SL_TOP, 3);
  ymir_cvec_t   *tmp = ymir_cvec_new (mesh, 3);
  ymir_vec_t    *up_inc = hessian_params->up_inc;
  ymir_cvec_t   *uinc = ymir_cvec_new (mesh, 3);
  ymir_cvec_t   *uout = hessian_params->uout;
  ymir_vec_t    *uout_inc = ymir_cvec_new (mesh, 3);
  ymir_vec_t    *upout_inc = hessian_params->upout_inc;
  ymir_vec_t    *tmp_surf = ymir_face_cvec_new (mesh, SL_TOP, 3);

  YMIR_GLOBAL_PRODUCTION ("entering hessian block uu.\n");
  ymir_upvec_get_u (up_inc, uinc, YMIR_READ);
  ymir_interp_vec (uinc, intrace);
  YMIR_GLOBAL_PRODUCTION ("interpolating.\n");
  ymir_upvec_set_u (up_inc, uinc, YMIR_RELEASE);
  YMIR_GLOBAL_PRODUCTION ("releasing.\n");

  ymir_interp_vec (intrace_mass, tmp);
  ymir_mass_apply (tmp, tmp_surf);
  ymir_interp_vec (tmp_surf, uout_inc);
  ymir_stokes_vec_set_velocity (uout_inc, upout_inc, press_elem);

  /* YMIR_GLOBAL_PRODUCTION ("setting vector.\n"); */
  
 
  ymir_vec_destroy (uinc);
  ymir_vec_destroy (intrace);
  ymir_vec_destroy (intrace_mass);
  ymir_vec_destroy (tmp);
  ymir_vec_destroy (tmp_surf);
  ymir_vec_destroy (uout_inc);
}
#endif

void
slabs_hessian_block_uu_newton (slabs_inverse_problem_params_t *inverse_params,
			       slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_pressure_elem_t *press_elem = inverse_params->press_elem;
  mangll_t             *mangll = mesh->ma;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *out = hessian_params->hessian_block_uu_newton_inc;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *viscosity_second_deriv_IIe = inverse_params->viscosity_second_deriv_IIe;
  ymir_vec_t           *eueu_inc = hessian_params->eueu_inc;
  ymir_vec_t           *euev = hessian_params->euev;
  ymir_vec_t           *euinc_ev = hessian_params->euinc_ev;
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *out1 = ymir_vec_template (viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (viscosity); 
  ymir_vec_t           *out3 = ymir_vec_template (viscosity); 
  ymir_vec_t           *u = hessian_params->uout;
  ymir_vec_t           *u_inc = hessian_params->u_inc;
  ymir_vec_t           *v = hessian_params->v;
  ymir_vec_t           *u_apply1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply3 = ymir_cvec_new (mesh, 3);
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *out1_el_mat, *out2_el_mat, *out3_el_mat, *euev_el_mat;
  sc_dmatrix_t       *eueu_inc_el_mat, *viscosity_deriv_IIe_el_mat, *viscosity_second_deriv_IIe_el_mat;
  sc_dmatrix_t       *euinc_ev_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_second_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  eueu_inc_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euinc_ev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out3_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (eueu_inc, eueu_inc_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euinc_ev, euinc_ev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_second_deriv_IIe, viscosity_second_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict eueuinc = eueu_inc_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_IIe_second_deriv = viscosity_second_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict eu_ev = euev_el_mat->e[0] + nodeid;
      double *_sc_restrict euev_inc = euinc_ev_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict output3 = out3_el_mat->e[0] +  nodeid;
 
      output1[0] = viscosity_IIe_second_deriv[0] * eueuinc[0] * eu_ev[0] + viscosity_IIe_deriv[0] * euev_inc[0];
      output2[0] = viscosity_IIe_deriv[0] * eu_ev[0];
      output3[0] = viscosity_IIe_deriv[0] * eueuinc[0];
      
      
                                                                                                                   
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  ymir_stress_op_t *stress_op_mod1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod2 = ymir_stress_op_new (out2, vel_dir, NULL, u_inc, NULL);
  ymir_stress_op_t *stress_op_mod3 = ymir_stress_op_new (out3, vel_dir, NULL, v, NULL);

  ymir_stress_op_apply (u, u_apply1, stress_op_mod1);
  ymir_stress_op_apply (u_inc, u_apply2, stress_op_mod2);
  ymir_stress_op_apply (v, u_apply3, stress_op_mod3);
  ymir_vec_add (1.0, u_apply1, u_apply2);
  ymir_vec_add (1.0, u_apply2, u_apply3);

  ymir_stokes_vec_set_velocity (u_apply3,  out, press_elem);
  ymir_stress_op_destroy (stress_op_mod1);
  ymir_stress_op_destroy (stress_op_mod2);
  ymir_stress_op_destroy (stress_op_mod3);

  

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (viscosity_second_deriv_IIe_el_mat);
  sc_dmatrix_destroy (eueu_inc_el_mat);
  sc_dmatrix_destroy (euinc_ev_el_mat);
  sc_dmatrix_destroy (euev_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);
  sc_dmatrix_destroy (out3_el_mat);
 
}


void
slabs_hessian_block_mm_strain_exp (slabs_inverse_problem_params_t *inverse_params,
				   slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *euev = inverse_params->euev;
  ymir_vec_t  *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t  *grad_viscosity_strain_rate_exponent = inverse_params->grad_viscosity_strain_rate_exponent;
  ymir_vec_t  *out = ymir_vec_clone (euev);
  ymir_vec_t  *hessian_block_mm_strain_rate_exponent = hessian_params->hessian_block_mm_strain_rate_exponent;

  ymir_vec_multiply_in1 (grad_viscosity_strain_rate_exponent, out);
  ymir_vec_copy (out, hessian_block_mm_strain_rate_exponent);

  ymir_vec_destroy (out);
}


void
slabs_hessian_block_mm_upper_mantle (slabs_inverse_problem_params_t *inverse_params,
				     slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *euev = inverse_params->euev;
  ymir_vec_t  *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t  *grad_viscosity_upper_mantle = inverse_params->grad_viscosity_UM_prefactor;
  ymir_vec_t  *hessian_block_mm_UM_prefactor = hessian_params->hessian_block_mm_UM_prefactor;
  ymir_vec_t  *out = ymir_vec_clone (euev);
  
  ymir_vec_multiply_in1 (grad_viscosity_upper_mantle, out);
  ymir_vec_copy (out, hessian_block_mm_UM_prefactor);

  ymir_vec_destroy (out);


}

void
slabs_hessian_block_mm_transition_zone (slabs_inverse_problem_params_t *inverse_params,
					slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *euev = inverse_params->euev;
  ymir_vec_t  *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_vec_t  *grad_viscosity_transition_zone = inverse_params->grad_viscosity_TZ_prefactor;
  ymir_vec_t  *hessian_block_mm_TZ_prefactor = hessian_params->hessian_block_mm_TZ_prefactor;
  ymir_vec_t  *out = ymir_vec_clone (euev);
  
  ymir_vec_multiply_in1 (grad_viscosity_transition_zone, out);
  ymir_vec_copy (out, hessian_block_mm_TZ_prefactor);

  ymir_vec_destroy (out);


}
				   

void
slabs_hessian_block_mm_weakfactor (slabs_inverse_problem_params_t *inverse_params,
				   slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *euev = inverse_params->euev;
  ymir_vec_t  *weakzone_marker = inverse_params->weakzone_marker;
  ymir_vec_t  *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t  *hessian_block_mm_weakfactor = hessian_params->hessian_block_mm_weakfactor;
  ymir_vec_t  *out = ymir_vec_clone (euev);
  
  ymir_vec_multiply_in1 (grad_viscosity_weakfactor, out);
  ymir_vec_copy (out, hessian_block_mm_weakfactor);

  ymir_vec_destroy (out);


}

void
slabs_hessian_block_mm_yield_stress (slabs_inverse_problem_params_t *inverse_params,
				     slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *euev = inverse_params->euev;
  ymir_vec_t  *yielding_marker = inverse_params->yielding_marker;
  ymir_vec_t  *grad_viscosity_yield_stress = inverse_params->grad_viscosity_yield_stress;
  ymir_vec_t  *hessian_block_mm_yield_stress = hessian_params->hessian_block_mm_yield_stress;
  ymir_vec_t  *out = ymir_vec_clone (euev);
  
  ymir_vec_multiply_in1 (grad_viscosity_yield_stress, out);
  ymir_vec_copy (out, hessian_block_mm_yield_stress);

  ymir_vec_destroy (out);


}


void
slabs_hessian_block_mm_strain_exp_prior (ymir_vec_t *strain_rate_exp_inc,
					 slabs_inverse_problem_params_t *inverse_params,
					 slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t  *prior_strain_rate_exp_misfit = inverse_params->prior_strain_rate_exp_misfit;
  ymir_vec_t  *strain_rate_exp_distrib = inverse_params->strain_rate_exp_distrib;
  ymir_vec_t  *out = ymir_vec_clone (upper_mantle_marker);

  
  ymir_vec_copy (prior_strain_rate_exp_misfit, strain_rate_exp_inc);
  ymir_vec_multiply_in1 (strain_rate_exp_distrib, strain_rate_exp_inc);
  ymir_vec_multiply_in1 (strain_rate_exp_distrib, out);
  ymir_vec_add (1.0, out, strain_rate_exp_inc);
  
  ymir_vec_destroy (out);

}


void
slabs_hessian_block_mm_weakfactor_prior (ymir_vec_t *weakfactor_inc,
					 slabs_inverse_problem_params_t *inverse_params,
					 slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *weakfactor_marker = inverse_params->weakzone_marker;
  ymir_vec_t  *prior_weakfactor_misfit = inverse_params->prior_weakzone_misfit;
  ymir_vec_t  *weakfactor_distrib = inverse_params->weakzone_distrib;
  ymir_vec_t  *out = ymir_vec_clone (weakfactor_marker);

  
  ymir_vec_copy (prior_weakfactor_misfit, weakfactor_inc);
  ymir_vec_multiply_in1 (weakfactor_distrib, weakfactor_inc);
  ymir_vec_multiply_in1 (weakfactor_distrib, out);
  ymir_vec_add (1.0, out,weakfactor_inc);
  
  ymir_vec_destroy (out);

}


void
slabs_hessian_block_mm_upper_mantle_prior (ymir_vec_t *upper_mantle_inc,
					   slabs_inverse_problem_params_t *inverse_params,
					   slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t  *prior_upper_mantle_misfit = inverse_params->prior_upper_mantle_misfit;
  ymir_vec_t  *upper_mantle_distrib = inverse_params->upper_mantle_distrib;
  ymir_vec_t  *out = ymir_vec_clone (upper_mantle_marker);

  
  ymir_vec_copy (prior_upper_mantle_misfit, upper_mantle_inc);
  ymir_vec_multiply_in1 (upper_mantle_distrib, upper_mantle_inc);
  ymir_vec_multiply_in1 (upper_mantle_distrib, out);
  ymir_vec_add (1.0, out, upper_mantle_inc);
  
  ymir_vec_destroy (out);



}


void
slabs_hessian_block_mm_transition_zone_prior (ymir_vec_t *transition_zone_inc,
					      slabs_inverse_problem_params_t *inverse_params,
					      slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_vec_t  *prior_transition_zone_misfit = inverse_params->prior_transition_zone_misfit;
  ymir_vec_t  *transition_zone_distrib = inverse_params->transition_zone_distrib;
  ymir_vec_t  *out = ymir_vec_clone (transition_zone_marker);

  
  ymir_vec_copy (prior_transition_zone_misfit, transition_zone_inc);
  ymir_vec_multiply_in1 (transition_zone_distrib, transition_zone_inc);
  ymir_vec_multiply_in1 (transition_zone_distrib, out);
  ymir_vec_add (1.0, out, transition_zone_inc);
  
  ymir_vec_destroy (out);



}


void
slabs_hessian_block_mm_lower_mantle_prior (ymir_vec_t *lower_mantle_inc,
					   slabs_inverse_problem_params_t *inverse_params,
					   slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *lower_mantle_marker = inverse_params->lower_mantle_marker;
  ymir_vec_t  *prior_lower_mantle_misfit = inverse_params->prior_lower_mantle_misfit;
  ymir_vec_t  *lower_mantle_distrib = inverse_params->lower_mantle_distrib;
  ymir_vec_t  *out = ymir_vec_clone (lower_mantle_marker);

  
  ymir_vec_copy (prior_lower_mantle_misfit, lower_mantle_inc);
  ymir_vec_multiply_in1 (lower_mantle_distrib, lower_mantle_inc);
  ymir_vec_multiply_in1 (lower_mantle_distrib, out);
  ymir_vec_add (1.0, out, lower_mantle_inc);
  
  ymir_vec_destroy (out);



}


void
slabs_hessian_block_mm_yield_stress_prior (ymir_vec_t *yield_stress_inc,
					 slabs_inverse_problem_params_t *inverse_params,
					 slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *ones = inverse_params->ones;
  ymir_vec_t  *prior_yield_stress_misfit = inverse_params->prior_yield_stress_misfit;
  ymir_vec_t  *yield_stress_distrib = inverse_params->yield_stress_distrib;
  ymir_vec_t  *out = ymir_vec_clone (ones);

  
  ymir_vec_copy (prior_yield_stress_misfit, yield_stress_inc);
  ymir_vec_multiply_in1 (yield_stress_distrib, yield_stress_inc);
  ymir_vec_multiply_in1 (yield_stress_distrib, out);
  ymir_vec_add (1.0, out, yield_stress_inc);
  
  ymir_vec_destroy (out);

}

void 
slabs_hessian_block_vm_prefactor (ymir_vec_t *vq_out,
				  slabs_inverse_problem_params_t *inverse_params)
				  
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc;
  ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem;
  ymir_cvec_t *u = stokes_pc->stokes_op->stress_op->usol;
  ymir_stokes_op_t *stokes_op = stokes_pc->stokes_op;
  ymir_vel_dir_t *vel_dir = inverse_params->vel_dir;
  ymir_vec_t *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t *rhs_inc = ymir_stokes_vec_new (mesh, press_elem);
  ymir_cvec_t *u_apply = ymir_vec_template (u);
  
  YMIR_GLOBAL_PRODUCTION ("entering hessian block vm.\n");
 
#if 1 /* Use the stiffness routines */
  ymir_vec_set_zero (rhs_inc);
  ymir_stress_op_t *stress_op_mod = ymir_stress_op_new (grad_viscosity_weakfactor, 
							vel_dir, NULL, u, NULL);

  ymir_stress_op_apply (u, u_apply, stress_op_mod);
  ymir_stokes_vec_set_velocity (u_apply, rhs_inc, press_elem);
  ymir_vec_copy (rhs_inc, vq_out);
  ymir_stress_op_destroy (stress_op_mod);
#endif

  YMIR_GLOBAL_PRODUCTION ("Computed RHS.\n");


  ymir_vec_destroy (rhs_inc);
  ymir_vec_destroy (u_apply);
  YMIR_GLOBAL_PRODUCTION ("Exiting Hessian VM_prefactor block.\n");


}

void 
slabs_hessian_block_vm_yield_stress (ymir_vec_t *vq_out,
				     slabs_inverse_problem_params_t *inverse_params)
  
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc;
  ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem;
  ymir_cvec_t *u = stokes_pc->stokes_op->stress_op->usol;
  ymir_stokes_op_t *stokes_op = stokes_pc->stokes_op;
  ymir_vel_dir_t *vel_dir = inverse_params->vel_dir;
  ymir_vec_t *grad_viscosity_yield_stress = inverse_params->grad_viscosity_yield_stress;
  ymir_vec_t *rhs_inc = ymir_stokes_vec_new (mesh, press_elem);
  ymir_cvec_t *u_apply = ymir_vec_template (u);
  
  YMIR_GLOBAL_PRODUCTION ("entering hessian block vm.\n");
 
#if 1 /* Use the stiffness routines */
  ymir_vec_set_zero (rhs_inc);
  ymir_stress_op_t *stress_op_mod = ymir_stress_op_new (grad_viscosity_yield_stress, 
							vel_dir, NULL, u, NULL);

  ymir_stress_op_apply (u, u_apply, stress_op_mod);
  ymir_stokes_vec_set_velocity (u_apply, rhs_inc, press_elem);
  ymir_vec_copy (rhs_inc, vq_out);
  ymir_stress_op_destroy (stress_op_mod);
#endif

  YMIR_GLOBAL_PRODUCTION ("Computed RHS.\n");


  ymir_vec_destroy (rhs_inc);
  ymir_vec_destroy (u_apply);
  YMIR_GLOBAL_PRODUCTION ("Exiting Hessian VM_prefactor block.\n");


}

void 
slabs_hessian_block_vm_yield_upper_mantle_prefactor (ymir_vec_t *vq_out,
						     slabs_inverse_problem_params_t *inverse_params)
  
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc;
  ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem;
  ymir_cvec_t *u = stokes_pc->stokes_op->stress_op->usol;
  ymir_stokes_op_t *stokes_op = stokes_pc->stokes_op;
  ymir_vel_dir_t *vel_dir = inverse_params->vel_dir;
  ymir_vec_t *grad_viscosity_upper_mantle = inverse_params->grad_viscosity_UM_prefactor;
  ymir_vec_t *rhs_inc = ymir_stokes_vec_new (mesh, press_elem);
  ymir_cvec_t *u_apply = ymir_vec_template (u);
  
  YMIR_GLOBAL_PRODUCTION ("entering hessian block vm.\n");
 
#if 1 /* Use the stiffness routines */
  ymir_vec_set_zero (rhs_inc);
  ymir_stress_op_t *stress_op_mod = ymir_stress_op_new (grad_viscosity_upper_mantle, 
							vel_dir, NULL, u, NULL);

  ymir_stress_op_apply (u, u_apply, stress_op_mod);
  ymir_stokes_vec_set_velocity (u_apply, rhs_inc, press_elem);
  ymir_vec_copy (rhs_inc, vq_out);
  ymir_stress_op_destroy (stress_op_mod);
#endif

  YMIR_GLOBAL_PRODUCTION ("Computed RHS.\n");


  ymir_vec_destroy (rhs_inc);
  ymir_vec_destroy (u_apply);
  YMIR_GLOBAL_PRODUCTION ("Exiting Hessian VM_prefactor block.\n");


}

void 
slabs_hessian_block_vm_yield_transition_zone_prefactor (ymir_vec_t *vq_out,
							slabs_inverse_problem_params_t *inverse_params)
  
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc;
  ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem;
  ymir_cvec_t *u = stokes_pc->stokes_op->stress_op->usol;
  ymir_stokes_op_t *stokes_op = stokes_pc->stokes_op;
  ymir_vel_dir_t *vel_dir = inverse_params->vel_dir;
  ymir_vec_t *grad_viscosity_transition_zone = inverse_params->grad_viscosity_TZ_prefactor;
  ymir_vec_t *rhs_inc = ymir_stokes_vec_new (mesh, press_elem);
  ymir_cvec_t *u_apply = ymir_vec_template (u);
  
  YMIR_GLOBAL_PRODUCTION ("entering hessian block vm.\n");
 
#if 1 /* Use the stiffness routines */
  ymir_vec_set_zero (rhs_inc);
  ymir_stress_op_t *stress_op_mod = ymir_stress_op_new (grad_viscosity_transition_zone, 
							vel_dir, NULL, u, NULL);

  ymir_stress_op_apply (u, u_apply, stress_op_mod);
  ymir_stokes_vec_set_velocity (u_apply, rhs_inc, press_elem);
  ymir_vec_copy (rhs_inc, vq_out);
  ymir_stress_op_destroy (stress_op_mod);
#endif

  YMIR_GLOBAL_PRODUCTION ("Computed RHS.\n");


  ymir_vec_destroy (rhs_inc);
  ymir_vec_destroy (u_apply);
  YMIR_GLOBAL_PRODUCTION ("Exiting Hessian VM_prefactor block.\n");

}

 
void 
slabs_hessian_block_vm_yield_activation_energy (ymir_vec_t *vq_out,
						     slabs_inverse_problem_params_t *inverse_params)
  
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc;
  ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem;
  ymir_cvec_t *u = stokes_pc->stokes_op->stress_op->usol;
  ymir_stokes_op_t *stokes_op = stokes_pc->stokes_op;
  ymir_vel_dir_t *vel_dir = inverse_params->vel_dir;
  ymir_vec_t *grad_viscosity_activation_energy = inverse_params->grad_viscosity_activation_energy;
  ymir_vec_t *rhs_inc = ymir_stokes_vec_new (mesh, press_elem);
  ymir_cvec_t *u_apply = ymir_vec_template (u);
  
  YMIR_GLOBAL_PRODUCTION ("entering hessian block vm.\n");
 
#if 1 /* Use the stiffness routines */
  ymir_vec_set_zero (rhs_inc);
  ymir_stress_op_t *stress_op_mod = ymir_stress_op_new (grad_viscosity_activation_energy, 
							vel_dir, NULL, u, NULL);

  ymir_stress_op_apply (u, u_apply, stress_op_mod);
  ymir_stokes_vec_set_velocity (u_apply, rhs_inc, press_elem);
  ymir_vec_copy (rhs_inc, vq_out);
  ymir_stress_op_destroy (stress_op_mod);
#endif

  YMIR_GLOBAL_PRODUCTION ("Computed RHS.\n");


  ymir_vec_destroy (rhs_inc);
  ymir_vec_destroy (u_apply);
  YMIR_GLOBAL_PRODUCTION ("Exiting Hessian VM_prefactor block.\n");


}

void 
slabs_hessian_block_vm_yield_strain_rate_exponent (ymir_vec_t *vq_out,
						   slabs_inverse_problem_params_t *inverse_params)
  
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc;
  ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem;
  ymir_cvec_t *u = stokes_pc->stokes_op->stress_op->usol;
  ymir_stokes_op_t *stokes_op = stokes_pc->stokes_op;
  ymir_vel_dir_t *vel_dir = inverse_params->vel_dir;
  ymir_vec_t *grad_viscosity_strain_rate_exponent = inverse_params->grad_viscosity_strain_rate_exponent;
  ymir_vec_t *rhs_inc = ymir_stokes_vec_new (mesh, press_elem);
  ymir_cvec_t *u_apply = ymir_vec_template (u);
  
  YMIR_GLOBAL_PRODUCTION ("entering hessian block vm.\n");
 
#if 1 /* Use the stiffness routines */
  ymir_vec_set_zero (rhs_inc);
  ymir_stress_op_t *stress_op_mod = ymir_stress_op_new (grad_viscosity_strain_rate_exponent, 
							vel_dir, NULL, u, NULL);

  ymir_stress_op_apply (u, u_apply, stress_op_mod);
  ymir_stokes_vec_set_velocity (u_apply, rhs_inc, press_elem);
  ymir_vec_copy (rhs_inc, vq_out);
  ymir_stress_op_destroy (stress_op_mod);
#endif

  YMIR_GLOBAL_PRODUCTION ("Computed RHS.\n");


  ymir_vec_destroy (rhs_inc);
  ymir_vec_destroy (u_apply);
  YMIR_GLOBAL_PRODUCTION ("Exiting Hessian VM_prefactor block.\n");

}


void 
slabs_hessian_block_mv_weakfactor (slabs_hessian_params_t *hessian_params,
				   slabs_inverse_problem_params_t *inverse_params)


{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *eu = inverse_params->eu;
  ymir_dvec_t *ev_inc = hessian_params->ev_inc;
  ymir_dvec_t *euev_inc = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_vec_t  *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_cvec_t *u = inverse_params->u;
  ymir_dvec_t *ones = inverse_params->ones;
  ymir_cvec_t *v_inc = hessian_params->v_inc;
  ymir_vec_t  *weakfactor_inc = hessian_params->weakfactor_inc;

  YMIR_GLOBAL_PRODUCTION ("Entering hessian block mv.\n");
  /* compute the inner-product of strain rate tensors e(u):e(v) */
  ymir_velocity_symtens_dotprod (eu, ev_inc, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("Constructed eu:ev.\n");



  ymir_dvec_multiply_in1 (grad_viscosity_weakfactor, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("multiply incremental.\n");
  
  ymir_mass_apply (euev_inc, weakfactor_inc);
  ymir_vec_destroy (euev_inc);


}



void 
slabs_hessian_block_mv_strain_rate_exponent (slabs_hessian_params_t *hessian_params,
					     slabs_inverse_problem_params_t *inverse_params)


{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *eu = inverse_params->eu;
  ymir_dvec_t *ev_inc = hessian_params->ev_inc;
  ymir_dvec_t *euev_inc = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_vec_t  *grad_viscosity_strain_rate_exponent = inverse_params->grad_viscosity_strain_rate_exponent;
  ymir_cvec_t *u = inverse_params->u;
  ymir_dvec_t *ones = inverse_params->ones;
  ymir_cvec_t *v_inc = hessian_params->v_inc;
  ymir_vec_t  *strain_rate_exp_inc = hessian_params->strain_rate_exp_inc;

  YMIR_GLOBAL_PRODUCTION ("Entering hessian block mv.\n");
  /* compute the inner-product of strain rate tensors e(u):e(v) */
  ymir_velocity_symtens_dotprod (eu, ev_inc, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("Constructed eu:ev.\n");

  ymir_dvec_multiply_in1 (grad_viscosity_strain_rate_exponent, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("multiply incremental.\n");
  
  ymir_mass_apply (euev_inc, strain_rate_exp_inc);
  ymir_vec_destroy (euev_inc);


}


void 
slabs_hessian_block_mv_yield_stress (slabs_hessian_params_t *hessian_params,
				     slabs_inverse_problem_params_t *inverse_params)


{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *eu = inverse_params->eu;
  ymir_dvec_t *ev_inc = hessian_params->ev_inc;
  ymir_dvec_t *euev_inc = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_vec_t  *grad_viscosity_yield_stress = inverse_params->grad_viscosity_yield_stress;
  ymir_cvec_t *u = inverse_params->u;
  ymir_dvec_t *ones = inverse_params->ones;
  ymir_cvec_t *v_inc = hessian_params->v_inc;
  ymir_vec_t  *yield_stress_inc = hessian_params->yield_stress_inc;

  YMIR_GLOBAL_PRODUCTION ("Entering hessian block mv.\n");
  /* compute the inner-product of strain rate tensors e(u):e(v) */
  ymir_velocity_symtens_dotprod (eu, ev_inc, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("Constructed eu:ev.\n");

  ymir_dvec_multiply_in1 (grad_viscosity_yield_stress, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("multiply incremental.\n");
  
  ymir_mass_apply (euev_inc, yield_stress_inc);
  ymir_vec_destroy (euev_inc);

}


void 
slabs_hessian_block_mv_upper_mantle_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *eu = inverse_params->eu;
  ymir_dvec_t *ev_inc = hessian_params->ev_inc;
  ymir_dvec_t *euev_inc = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_vec_t  *grad_viscosity_upper_mantle = inverse_params->grad_viscosity_UM_prefactor;
  ymir_cvec_t *u = inverse_params->u;
  ymir_dvec_t *ones = inverse_params->ones;
  ymir_cvec_t *v_inc = hessian_params->v_inc;
  ymir_vec_t  *upper_mantle_prefactor_inc = hessian_params->upper_mantle_prefactor_inc;

  YMIR_GLOBAL_PRODUCTION ("Entering hessian block mv.\n");
  /* compute the inner-product of strain rate tensors e(u):e(v) */
  ymir_velocity_symtens_dotprod (eu, ev_inc, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("Constructed eu:ev.\n");

  ymir_dvec_multiply_in1 (grad_viscosity_upper_mantle, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("multiply incremental.\n");
  
  ymir_mass_apply (euev_inc, upper_mantle_prefactor_inc);
  ymir_vec_destroy (euev_inc);


}

void 
slabs_hessian_block_mv_transition_zone_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *eu = inverse_params->eu;
  ymir_dvec_t *ev_inc = hessian_params->ev_inc;
  ymir_dvec_t *euev_inc = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_vec_t  *grad_viscosity_transition_zone = inverse_params->grad_viscosity_TZ_prefactor;
  ymir_cvec_t *u = inverse_params->u;
  ymir_dvec_t *ones = inverse_params->ones;
  ymir_cvec_t *v_inc = hessian_params->v_inc;
  ymir_vec_t  *transition_zone_prefactor_inc = hessian_params->transition_zone_prefactor_inc;

  YMIR_GLOBAL_PRODUCTION ("Entering hessian block mv.\n");
  /* compute the inner-product of strain rate tensors e(u):e(v) */
  ymir_velocity_symtens_dotprod (eu, ev_inc, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("Constructed eu:ev.\n");

  ymir_dvec_multiply_in1 (grad_viscosity_transition_zone, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("multiply incremental.\n");
  
  ymir_mass_apply (euev_inc, transition_zone_prefactor_inc);
  ymir_vec_destroy (euev_inc);


}


void 
slabs_hessian_block_mv_activation_energy (slabs_hessian_params_t *hessian_params,
					  slabs_inverse_problem_params_t *inverse_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *eu = inverse_params->eu;
  ymir_dvec_t *ev_inc = hessian_params->ev_inc;
  ymir_dvec_t *euev_inc = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_vec_t  *grad_viscosity_activation_energy = inverse_params->grad_viscosity_activation_energy;
  ymir_cvec_t *u = inverse_params->u;
  ymir_dvec_t *ones = inverse_params->ones;
  ymir_cvec_t *v_inc = hessian_params->v_inc;
  ymir_vec_t  *activation_energy_inc = hessian_params->activation_energy_inc;

  YMIR_GLOBAL_PRODUCTION ("Entering hessian block mv.\n");
  /* compute the inner-product of strain rate tensors e(u):e(v) */
  ymir_velocity_symtens_dotprod (eu, ev_inc, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("Constructed eu:ev.\n");

  ymir_dvec_multiply_in1 (grad_viscosity_activation_energy, euev_inc);
  YMIR_GLOBAL_PRODUCTION ("multiply incremental.\n");
  
  ymir_mass_apply (euev_inc, activation_energy_inc);
  ymir_vec_destroy (euev_inc);


}


void 
slabs_hessian_block_um_weakfactor (slabs_hessian_params_t *hessian_params,
				   slabs_inverse_problem_params_t *inverse_params,
				   const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t           *grad_weak_factor = inverse_params->grad_weak_factor;
  ymir_vec_t           *weakzone_marker = inverse_params->weakzone_marker;
  ymir_vec_t           *out1 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  ymir_vec_t           *rhs1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *rhs2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *v = inverse_params->adjoint_v;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out1_el_mat, *out2_el_mat;
  sc_dmatrix_t       *grad_viscosity_weakfactor_el_mat, *weakzone_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *grad_weak_factor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_weakfactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  weakzone_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_weak_factor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (weakzone_marker, weakzone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_weak_factor, grad_weak_factor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_weak_factor, grad_weak_factor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);


    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_weakfactor = grad_viscosity_weakfactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict eu_ev_inner = euev_el_mat->e[0] + nodeid; 
      double *_sc_restrict gradient_weakfactor = grad_weak_factor_el_mat->e[0] + nodeid;
      output1[0] = grad_viscosity_weakfactor[0] * eu_ev_inner[0];
      output2[0] = gradient_weakfactor[0];

    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }
  ymir_stress_op_t *stress_op1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op2 = ymir_stress_op_new (out2, vel_dir, NULL, v, NULL);
  
  ymir_stress_op_apply (u, rhs1, stress_op1);
  ymir_stress_op_apply (v, rhs2, stress_op2);
  ymir_vec_add (1.0, rhs1, rhs2);
  
  ymir_stress_op_destroy (stress_op1);
  ymir_stress_op_destroy (stress_op2);

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (weakzone_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_weakfactor_el_mat);
  sc_dmatrix_destroy (grad_weak_factor_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);

}


void 
slabs_hessian_block_um_strain_rate_exponent (slabs_hessian_params_t *hessian_params,
					     slabs_inverse_problem_params_t *inverse_params,
					     const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_strain_rate_exponent = inverse_params->grad_viscosity_strain_rate_exponent;
  ymir_vec_t           *grad_strain_exp = inverse_params->grad_strain_exp;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out1 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  ymir_vec_t           *rhs1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *rhs2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *v = inverse_params->adjoint_v;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  mangll_t             *mangll = mesh->ma;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out1_el_mat, *out2_el_mat;
  sc_dmatrix_t       *grad_viscosity_strain_rate_exponent_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *grad_strain_exp_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_strain_rate_exponent_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_strain_exp_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_strain_rate_exponent, grad_viscosity_strain_rate_exponent_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_strain_exp, grad_strain_exp_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);

    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_strain_exp = grad_viscosity_strain_rate_exponent_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict eu_ev_inner = euev_el_mat->e[0] + nodeid; 
      double *_sc_restrict gradient_strain_rate_exp = grad_strain_exp_el_mat->e[0] + nodeid;
      output1[0] = grad_viscosity_strain_exp[0] * eu_ev_inner[0];
      output2[0] = gradient_strain_rate_exp[0];

      //      output[0] = 
    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_stress_op_t *stress_op1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op2 = ymir_stress_op_new (out2, vel_dir, NULL, v, NULL);
  
  ymir_stress_op_apply (u, rhs1, stress_op1);
  ymir_stress_op_apply (v, rhs2, stress_op2);
  ymir_vec_add (1.0, rhs1, rhs2);
  
  ymir_stress_op_destroy (stress_op1);
  ymir_stress_op_destroy (stress_op2);


  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */


 
  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_strain_rate_exponent_el_mat);
  sc_dmatrix_destroy (grad_strain_exp_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);

}


void 
slabs_hessian_block_um_yield_stress (slabs_hessian_params_t *hessian_params,
				     slabs_inverse_problem_params_t *inverse_params,
				     const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_yield_stress = inverse_params->grad_viscosity_yield_stress;
  ymir_vec_t           *grad_yield_stress = inverse_params->grad_yield_stress;
  ymir_vec_t           *yielding_marker = inverse_params->yielding_marker;
  ymir_vec_t           *out1 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  ymir_vec_t           *rhs1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *rhs2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *v = inverse_params->adjoint_v;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  mangll_t             *mangll = mesh->ma;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out1_el_mat, *out2_el_mat;
  sc_dmatrix_t       *grad_viscosity_yield_stress_el_mat, *yielding_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *grad_yield_stress_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  yielding_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_yield_stress_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_yield_stress_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (yielding_marker, yielding_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_yield_stress, grad_viscosity_yield_stress_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_yield_stress, grad_yield_stress_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);

    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_yield = grad_viscosity_yield_stress_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict eu_ev_inner = euev_el_mat->e[0] + nodeid; 
      double *_sc_restrict gradient_yield = grad_yield_stress_el_mat->e[0] + nodeid;
      output1[0] = grad_viscosity_yield[0] * eu_ev_inner[0];
      output2[0] = gradient_yield[0];

      //      output[0] = 
    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_stress_op_t *stress_op1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op2 = ymir_stress_op_new (out2, vel_dir, NULL, v, NULL);
  
  ymir_stress_op_apply (u, rhs1, stress_op1);
  ymir_stress_op_apply (v, rhs2, stress_op2);
  ymir_vec_add (1.0, rhs1, rhs2);
  
  ymir_stress_op_destroy (stress_op1);
  ymir_stress_op_destroy (stress_op2);


  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (yielding_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_yield_stress_el_mat);
  sc_dmatrix_destroy (grad_yield_stress_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);

}



void 
slabs_hessian_block_um_activation_energy (slabs_hessian_params_t *hessian_params,
					  slabs_inverse_problem_params_t *inverse_params,
					  const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t           *grad_weak_factor = inverse_params->grad_weak_factor;
  ymir_vec_t           *weakzone_marker = inverse_params->weakzone_marker;
  ymir_vec_t           *out = hessian_params->weakfactor_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_weakfactor_el_mat, *weakzone_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *grad_weak_factor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_weakfactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  weakzone_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_weak_factor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (weakzone_marker, weakzone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_weak_factor, grad_weak_factor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_weakfactor = grad_viscosity_weakfactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      //      output[0] = 
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (weakzone_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_weakfactor_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}


void 
slabs_hessian_block_um_upper_mantle_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params,
					       const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_UM_prefactor = inverse_params->grad_viscosity_UM_prefactor;
  ymir_vec_t           *grad_UM_prefactor = inverse_params->grad_UM_prefactor;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out1 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  ymir_vec_t           *rhs1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *rhs2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *v = inverse_params->adjoint_v;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  mangll_t             *mangll = mesh->ma;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out1_el_mat, *out2_el_mat;
  sc_dmatrix_t       *grad_viscosity_UM_prefactor_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *grad_UM_prefactor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_UM_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_UM_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_UM_prefactor, grad_viscosity_UM_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_UM_prefactor, grad_UM_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);

    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_UM = grad_viscosity_UM_prefactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict eu_ev_inner = euev_el_mat->e[0] + nodeid; 
      double *_sc_restrict gradient_UM = grad_UM_prefactor_el_mat->e[0] + nodeid;
      output1[0] = grad_viscosity_UM[0] * eu_ev_inner[0];
      output2[0] = gradient_UM[0];

      //      output[0] = 
    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_stress_op_t *stress_op1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op2 = ymir_stress_op_new (out2, vel_dir, NULL, v, NULL);
  
  ymir_stress_op_apply (u, rhs1, stress_op1);
  ymir_stress_op_apply (v, rhs2, stress_op2);
  ymir_vec_add (1.0, rhs1, rhs2);
  
  ymir_stress_op_destroy (stress_op1);
  ymir_stress_op_destroy (stress_op2);


  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_UM_prefactor_el_mat);
  sc_dmatrix_destroy (grad_UM_prefactor_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);

}

void 
slabs_hessian_block_um_transition_zone_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params,
					       const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_TZ_prefactor = inverse_params->grad_viscosity_TZ_prefactor;
  ymir_vec_t           *grad_TZ_prefactor = inverse_params->grad_TZ_prefactor;
  ymir_vec_t           *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_vec_t           *out1 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  ymir_vec_t           *rhs1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *rhs2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *v = inverse_params->adjoint_v;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  mangll_t             *mangll = mesh->ma;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out1_el_mat, *out2_el_mat;
  sc_dmatrix_t       *grad_viscosity_TZ_prefactor_el_mat, *transition_zone_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *grad_TZ_prefactor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_TZ_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_TZ_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (transition_zone_marker, transition_zone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_TZ_prefactor, grad_viscosity_TZ_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_TZ_prefactor, grad_TZ_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);

    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_TZ = grad_viscosity_TZ_prefactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict eu_ev_inner = euev_el_mat->e[0] + nodeid; 
      double *_sc_restrict gradient_TZ = grad_TZ_prefactor_el_mat->e[0] + nodeid;
      output1[0] = grad_viscosity_TZ[0] * eu_ev_inner[0];
      output2[0] = gradient_TZ[0];

      //      output[0] = 
    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_stress_op_t *stress_op1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op2 = ymir_stress_op_new (out2, vel_dir, NULL, v, NULL);
  
  ymir_stress_op_apply (u, rhs1, stress_op1);
  ymir_stress_op_apply (v, rhs2, stress_op2);
  ymir_vec_add (1.0, rhs1, rhs2);
  
  ymir_stress_op_destroy (stress_op1);
  ymir_stress_op_destroy (stress_op2);


  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (transition_zone_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_TZ_prefactor_el_mat);
  sc_dmatrix_destroy (grad_TZ_prefactor_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);

}


void 
slabs_hessian_block_mu_weakfactor (slabs_hessian_params_t *hessian_params,
				   slabs_inverse_problem_params_t *inverse_params,
				   const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t           *grad_weak_factor = inverse_params->grad_weak_factor;
  ymir_vec_t           *weakzone_marker = inverse_params->weakzone_marker;
  ymir_vec_t           *out1 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  ymir_vec_t           *euinc_ev = hessian_params->euinc_ev;
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *v = inverse_params->adjoint_v;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out1_el_mat, *out2_el_mat;
  sc_dmatrix_t       *grad_viscosity_weakfactor_el_mat, *weakzone_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *euinc_ev_el_mat, *grad_weak_factor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_weakfactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  weakzone_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euinc_ev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_weak_factor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (weakzone_marker, weakzone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_weak_factor, grad_weak_factor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euinc_ev, euinc_ev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_weak_factor, grad_weak_factor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);


    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_weakfactor = grad_viscosity_weakfactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict eu_ev_inner = euev_el_mat->e[0] + nodeid; 
      double *_sc_restrict euinc_ev_inner = euinc_ev_el_mat->e[0] + nodeid; 
      double *_sc_restrict gradient_weakfactor = grad_weak_factor_el_mat->e[0] + nodeid;
      output1[0] = grad_viscosity_weakfactor[0] * eu_ev_inner[0];
      output2[0] = gradient_weakfactor[0] * euinc_ev_inner[0];

    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }
  
  ymir_vec_add (1.0, out1, out2);
  

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (weakzone_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_weakfactor_el_mat);
  sc_dmatrix_destroy (grad_weak_factor_el_mat);
  sc_dmatrix_destroy (euev_el_mat);
  sc_dmatrix_destroy (euinc_ev_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);

}


void 
slabs_hessian_block_mu_strain_rate_exponent (slabs_hessian_params_t *hessian_params,
					     slabs_inverse_problem_params_t *inverse_params,
					     const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_strain_rate_exponent = inverse_params->grad_viscosity_strain_rate_exponent;
  ymir_vec_t           *grad_strain_exp = inverse_params->grad_strain_exp;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out1 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  ymir_vec_t           *euinc_ev = hessian_params->euinc_ev;
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *v = inverse_params->adjoint_v;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  mangll_t             *mangll = mesh->ma;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out1_el_mat, *out2_el_mat;
  sc_dmatrix_t       *grad_viscosity_strain_rate_exponent_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *euinc_ev_el_mat, *grad_strain_exp_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_strain_rate_exponent_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euinc_ev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_strain_exp_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_strain_rate_exponent, grad_viscosity_strain_rate_exponent_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_strain_exp, grad_strain_exp_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euinc_ev, euinc_ev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_strain_exp = grad_viscosity_strain_rate_exponent_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict eu_ev_inner = euev_el_mat->e[0] + nodeid; 
      double *_sc_restrict euinc_ev_inner = euinc_ev_el_mat->e[0] + nodeid; 
      double *_sc_restrict gradient_strain_rate_exp = grad_strain_exp_el_mat->e[0] + nodeid;
      output1[0] = grad_viscosity_strain_exp[0] * eu_ev_inner[0];
      output2[0] = gradient_strain_rate_exp[0] * euinc_ev_inner[0];

      //      output[0] = 
    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_vec_add (1.0, out1, out2);
  
  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_strain_rate_exponent_el_mat);
  sc_dmatrix_destroy (grad_strain_exp_el_mat);
  sc_dmatrix_destroy (euev_el_mat);
  sc_dmatrix_destroy (euinc_ev_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);

}


void 
slabs_hessian_block_mu_upper_mantle_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params,
					       const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_UM_prefactor = inverse_params->grad_viscosity_UM_prefactor;
  ymir_vec_t           *grad_UM_prefactor = inverse_params->grad_UM_prefactor;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out1 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  ymir_vec_t           *euinc_ev = hessian_params->euinc_ev;
  ymir_vec_t           *rhs1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *rhs2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *v = inverse_params->adjoint_v;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  mangll_t             *mangll = mesh->ma;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out1_el_mat, *out2_el_mat;
  sc_dmatrix_t       *grad_viscosity_UM_prefactor_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *euinc_ev_el_mat, *grad_UM_prefactor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_UM_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euinc_ev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_UM_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_UM_prefactor, grad_viscosity_UM_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_UM_prefactor, grad_UM_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euinc_ev, euinc_ev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_UM = grad_viscosity_UM_prefactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict eu_ev_inner = euev_el_mat->e[0] + nodeid; 
      double *_sc_restrict euinc_ev_inner = euinc_ev_el_mat->e[0] + nodeid; 
      double *_sc_restrict gradient_UM = grad_UM_prefactor_el_mat->e[0] + nodeid;
      output1[0] = grad_viscosity_UM[0] * eu_ev_inner[0];
      output2[0] = gradient_UM[0] * euinc_ev_inner[0];

      //      output[0] = 
    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_vec_add (1.0, out1, out2);


  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_UM_prefactor_el_mat);
  sc_dmatrix_destroy (grad_UM_prefactor_el_mat);
  sc_dmatrix_destroy (euev_el_mat);
  sc_dmatrix_destroy (euinc_ev_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);

}


void 
slabs_hessian_block_mu_transition_zone_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params,
					       const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_TZ_prefactor = inverse_params->grad_viscosity_TZ_prefactor;
  ymir_vec_t           *grad_TZ_prefactor = inverse_params->grad_TZ_prefactor;
  ymir_vec_t           *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_vec_t           *out1 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (inverse_params->viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *euev = inverse_params->euev;
  ymir_vec_t           *euinc_ev = hessian_params->euinc_ev;
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *v = inverse_params->adjoint_v;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  mangll_t             *mangll = mesh->ma;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out1_el_mat, *out2_el_mat;
  sc_dmatrix_t       *grad_viscosity_TZ_prefactor_el_mat, *transition_zone_marker_el_mat;
  sc_dmatrix_t       *euev_el_mat, *euinc_ev_el_mat, *grad_TZ_prefactor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_TZ_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  euinc_ev_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_TZ_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (transition_zone_marker, transition_zone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_TZ_prefactor, grad_viscosity_TZ_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_TZ_prefactor, grad_TZ_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euev, euev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (euinc_ev, euinc_ev_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_TZ = grad_viscosity_TZ_prefactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict eu_ev_inner = euev_el_mat->e[0] + nodeid; 
      double *_sc_restrict euinc_ev_inner = euinc_ev_el_mat->e[0] + nodeid; 
      double *_sc_restrict gradient_TZ = grad_TZ_prefactor_el_mat->e[0] + nodeid;
      output1[0] = grad_viscosity_TZ[0] * eu_ev_inner[0];
      output2[0] = gradient_TZ[0] * euinc_ev_inner[0];

      //      output[0] = 
    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_vec_add (1.0, out1, out2);

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (transition_zone_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_TZ_prefactor_el_mat);
  sc_dmatrix_destroy (grad_TZ_prefactor_el_mat);
  sc_dmatrix_destroy (euev_el_mat);
  sc_dmatrix_destroy (euinc_ev_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);

}



void 
slabs_hessian_block_uu_viscosity_average (slabs_hessian_params_t *hessian_params,
					  slabs_inverse_problem_params_t *inverse_params,
					  const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *out = hessian_params->hessian_block_uu_average_viscosity_inc;
  ymir_vec_t           *out1 = ymir_vec_template (viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (viscosity);
  ymir_vec_t           *out3 = ymir_vec_template (viscosity);
  ymir_vec_t           *out4 = ymir_vec_template (viscosity);
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *viscosity_second_deriv_IIe = inverse_params->viscosity_second_deriv_IIe;
  ymir_vec_t           *eueu_inc = hessian_params->eueu_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *u = inverse_params->u;
  ymir_vec_t           *u_inc = hessian_params->u_inc;
  ymir_vec_t           *u_apply1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply3 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply4 = ymir_cvec_new (mesh, 3);
  ymir_pressure_elem_t *press_elem = inverse_params->press_elem;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat;
  sc_dmatrix_t       *eueu_inc_el_mat, *viscosity_deriv_IIe_el_mat, *viscosity_second_deriv_IIe_el_mat;
  sc_dmatrix_t       *out1_el_mat, *out2_el_mat, *out3_el_mat, *out4_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_second_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  eueu_inc_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out3_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out4_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

   
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (eueu_inc, eueu_inc_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_second_deriv_IIe, viscosity_second_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out4, out4_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);


    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict eueuinc = eueu_inc_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_IIe_second_deriv = viscosity_second_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict output3 = out3_el_mat->e[0] +  nodeid;
      double *_sc_restrict output4 = out4_el_mat->e[0] +  nodeid;

      output1[0] = viscosity_IIe_deriv[0] * viscosity_IIe_deriv[0] * eueuinc[0] / (visc[0] * visc[0]);
      output2[0] = average_visc_area_misfit * viscosity_IIe_second_deriv[0] * eueuinc[0] / visc[0];
      output3[0] = -average_visc_area_misfit * viscosity_IIe_deriv[0] * viscosity_IIe_deriv[0] * eueuinc[0] / (visc[0] * visc[0]);
      output4[0] = average_visc_area_misfit * viscosity_IIe_deriv[0] * eueuinc[0] / visc[0];


    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
    ymir_dvec_set_elem (out4, out4_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  ymir_stress_op_t *stress_op_mod1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod2 = ymir_stress_op_new (out2, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod3 = ymir_stress_op_new (out3, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod4 = ymir_stress_op_new (out4, vel_dir, NULL, u_inc, NULL);

  ymir_stress_op_apply (u, u_apply1, stress_op_mod1);
  ymir_stress_op_apply (u, u_apply2, stress_op_mod2);
  ymir_stress_op_apply (u, u_apply3, stress_op_mod3);
  ymir_stress_op_apply (u_inc, u_apply4, stress_op_mod4);

  ymir_vec_add (1.0, u_apply1, u_apply2);
  ymir_vec_add (1.0, u_apply2, u_apply3);
  ymir_vec_add (1.0, u_apply3, u_apply4);

  ymir_stokes_vec_set_velocity (u_apply4,  out, press_elem);
  /* ymir_vec_copy (rhs_inc, vq_out); */
  ymir_stress_op_destroy (stress_op_mod1);
  ymir_stress_op_destroy (stress_op_mod2);
  ymir_stress_op_destroy (stress_op_mod3);
  ymir_stress_op_destroy (stress_op_mod4);


  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (viscosity_second_deriv_IIe_el_mat);
  sc_dmatrix_destroy (eueu_inc_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);
  sc_dmatrix_destroy (out3_el_mat);
  sc_dmatrix_destroy (out4_el_mat);
 
  ymir_vec_destroy (out1);
  ymir_vec_destroy (out2);
  ymir_vec_destroy (out3);
  ymir_vec_destroy (out4);
  ymir_vec_destroy (u_apply1);
  ymir_vec_destroy (u_apply2);
  ymir_vec_destroy (u_apply3);
  ymir_vec_destroy (u_apply4);

}


void 
slabs_hessian_block_mm_weakfactor_viscosity_average (slabs_hessian_params_t *hessian_params,
						     slabs_inverse_problem_params_t *inverse_params,
						     const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t           *weakzone_marker = inverse_params->weakzone_marker;
  ymir_vec_t           *out = hessian_params->weakfactor_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_weakfactor_el_mat, *weakzone_marker_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_weakfactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  weakzone_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (weakzone_marker, weakzone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_weakfactor = grad_viscosity_weakfactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * grad_viscosity_weakfactor[0] * grad_viscosity_weakfactor[0]/ (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_weakfactor[0] / visc[0]) -(average_visc_area_misfit * grad_viscosity_weakfactor[0] * grad_viscosity_weakfactor[0] / (visc[0] * visc[0]));
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (weakzone_marker_el_mat);
  sc_dmatrix_destroy (grad_viscosity_weakfactor_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}


void 
slabs_hessian_block_mm_strain_rate_exponent_viscosity_average (slabs_hessian_params_t *hessian_params,
							       slabs_inverse_problem_params_t *inverse_params,
							       const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *grad_viscosity_strain_rate_exponent = inverse_params->grad_viscosity_strain_rate_exponent;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out = hessian_params->strain_rate_exponent_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_strain_rate_exponent_el_mat, *upper_mantle_marker_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_strain_rate_exponent_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_strain_rate_exponent, grad_viscosity_strain_rate_exponent_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_strain_rate_exponent = grad_viscosity_strain_rate_exponent_el_mat->e[0] +  nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * grad_viscosity_strain_rate_exponent[0] * grad_viscosity_strain_rate_exponent[0]/ (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_strain_rate_exponent[0] / visc[0]) -(average_visc_area_misfit * grad_viscosity_strain_rate_exponent[0] * grad_viscosity_strain_rate_exponent[0] / (visc[0] * visc[0]));
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (grad_viscosity_strain_rate_exponent_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}

void 
slabs_hessian_block_mm_upper_mantle_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								 slabs_inverse_problem_params_t *inverse_params,
								 const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *grad_viscosity_upper_mantle = inverse_params->grad_viscosity_UM_prefactor;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out = hessian_params->upper_mantle_prefactor_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_upper_mantle_el_mat, *upper_mantle_marker_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_upper_mantle_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_upper_mantle, grad_viscosity_upper_mantle_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_upper_mantle = grad_viscosity_upper_mantle_el_mat->e[0] +  nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * grad_viscosity_upper_mantle[0] * grad_viscosity_upper_mantle[0]/ (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_upper_mantle[0] / visc[0]) -(average_visc_area_misfit * grad_viscosity_upper_mantle[0] * grad_viscosity_upper_mantle[0] / (visc[0] * visc[0]));
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (grad_viscosity_upper_mantle_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}

void 
slabs_hessian_block_mm_transition_zone_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								 slabs_inverse_problem_params_t *inverse_params,
								 const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *grad_viscosity_transition_zone = inverse_params->grad_viscosity_TZ_prefactor;
  ymir_vec_t           *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_vec_t           *out = hessian_params->transition_zone_prefactor_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_transition_zone_el_mat, *transition_zone_marker_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_transition_zone_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  transition_zone_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (transition_zone_marker, transition_zone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_transition_zone, grad_viscosity_transition_zone_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_transition_zone = grad_viscosity_transition_zone_el_mat->e[0] +  nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * grad_viscosity_transition_zone[0] * grad_viscosity_transition_zone[0]/ (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_transition_zone[0] / visc[0]) -(average_visc_area_misfit * grad_viscosity_transition_zone[0] * grad_viscosity_transition_zone[0] / (visc[0] * visc[0]));
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (transition_zone_marker_el_mat);
  sc_dmatrix_destroy (grad_viscosity_transition_zone_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}


void 
slabs_hessian_block_mu_weakfactor_viscosity_average (slabs_hessian_params_t *hessian_params,
						     slabs_inverse_problem_params_t *inverse_params,
						     const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_weakfactor = inverse_params->grad_viscosity_second_deriv_IIe_weakfactor;
  ymir_vec_t           *weakzone_marker = inverse_params->weakzone_marker;
  ymir_vec_t           *out = hessian_params->weakfactor_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *eueu_inc = hessian_params->eueu_inc;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_weakfactor_el_mat, *weakzone_marker_el_mat;
  sc_dmatrix_t       *eueu_inc_el_mat, *grad_viscosity_second_deriv_IIe_weakfactor_el_mat;
  sc_dmatrix_t       *out1_el_mat, *out2_el_mat, *out3_el_mat; 
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_weakfactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_weakfactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  weakzone_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (weakzone_marker, weakzone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_weakfactor, grad_viscosity_second_deriv_IIe_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (eueu_inc, eueu_inc_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_weakfactor = grad_viscosity_weakfactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_weak = grad_viscosity_second_deriv_IIe_weakfactor_el_mat->e[0] + nodeid;     
      double *_sc_restrict eueuinc = eueu_inc_el_mat->e[0] + nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * viscosity_IIe_deriv[0] * eueuinc[0] * grad_viscosity_weakfactor[0] / (visc[0] * visc[0])) - (average_visc_area_misfit * eueuinc[0] * viscosity_IIe_deriv[0] * grad_viscosity_weakfactor[0] / (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_weakfactor[0] * viscosity_IIe_deriv[0] * eueuinc[0] / visc[0]);
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (weakzone_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_weakfactor_el_mat);
  sc_dmatrix_destroy (eueu_inc_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}


void 
slabs_hessian_block_mu_strain_rate_exponent_viscosity_average (slabs_hessian_params_t *hessian_params,
							       slabs_inverse_problem_params_t *inverse_params,
							       const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_strain_rate_exponent = inverse_params->grad_viscosity_strain_rate_exponent;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_strain_rate_exponent = inverse_params->grad_viscosity_second_deriv_IIe_strain_rate_exponent;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out = hessian_params->strain_rate_exponent_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *eueu_inc = hessian_params->eueu_inc;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_strain_rate_exponent_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *eueu_inc_el_mat, *grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_strain_rate_exponent_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_strain_rate_exponent, grad_viscosity_strain_rate_exponent_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_strain_rate_exponent, grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (eueu_inc, eueu_inc_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_strain_rate_exponent = grad_viscosity_strain_rate_exponent_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_strain_exp = grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict eueuinc = eueu_inc_el_mat->e[0] + nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * viscosity_IIe_deriv[0] * eueuinc[0] * grad_viscosity_strain_rate_exponent[0] / (visc[0] * visc[0])) - (average_visc_area_misfit * eueuinc[0] * viscosity_IIe_deriv[0] * grad_viscosity_strain_rate_exponent[0] / (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_strain_rate_exponent[0] * viscosity_second_deriv_IIe_strain_exp[0] * eueuinc[0] / visc[0]);
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_strain_rate_exponent_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat);
  sc_dmatrix_destroy (eueu_inc_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}

void 
slabs_hessian_block_mu_upper_mantle_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								 slabs_inverse_problem_params_t *inverse_params,
								 const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_upper_mantle = inverse_params->grad_viscosity_UM_prefactor;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_UM_prefactor = inverse_params->grad_viscosity_second_deriv_IIe_UM_prefactor;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out = hessian_params->upper_mantle_prefactor_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *eueu_inc = hessian_params->eueu_inc;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_upper_mantle_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *eueu_inc_el_mat, *grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_upper_mantle_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_UM_prefactor, grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_upper_mantle, grad_viscosity_upper_mantle_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (eueu_inc, eueu_inc_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_upper_mantle = grad_viscosity_upper_mantle_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_UM = grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat->e[0] + nodeid;
      double *_sc_restrict eueuinc = eueu_inc_el_mat->e[0] + nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * viscosity_IIe_deriv[0] * eueuinc[0] * grad_viscosity_upper_mantle[0] / (visc[0] * visc[0])) - (average_visc_area_misfit * eueuinc[0] * viscosity_IIe_deriv[0] * grad_viscosity_upper_mantle[0] / (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_upper_mantle[0] * viscosity_second_deriv_IIe_UM[0] * eueuinc[0] / visc[0]);
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_upper_mantle_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat);
  sc_dmatrix_destroy (eueu_inc_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}


void 
slabs_hessian_block_mu_transition_zone_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								    slabs_inverse_problem_params_t *inverse_params,
								    const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_transition_zone = inverse_params->grad_viscosity_TZ_prefactor;
  ymir_vec_t           *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_TZ_prefactor = inverse_params->grad_viscosity_second_deriv_IIe_TZ_prefactor;
  ymir_vec_t           *out = hessian_params->transition_zone_prefactor_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *eueu_inc = hessian_params->eueu_inc;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_transition_zone_el_mat, *transition_zone_marker_el_mat;
  sc_dmatrix_t       *eueu_inc_el_mat, *grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat;
  double             *x, *y, *z, *tmp_el;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_transition_zone_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  transition_zone_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (transition_zone_marker, transition_zone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_transition_zone, grad_viscosity_transition_zone_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_TZ_prefactor, grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (eueu_inc, eueu_inc_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_transition_zone = grad_viscosity_transition_zone_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_TZ = grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat->e[0] + nodeid;
      double *_sc_restrict eueuinc = eueu_inc_el_mat->e[0] + nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * viscosity_IIe_deriv[0] * eueuinc[0] * grad_viscosity_transition_zone[0] / (visc[0] * visc[0])) - (average_visc_area_misfit * eueuinc[0] * viscosity_IIe_deriv[0] * grad_viscosity_transition_zone[0] / (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_transition_zone[0] * viscosity_second_deriv_IIe_TZ[0] * eueuinc[0] / visc[0]);
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (transition_zone_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_transition_zone_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat);
  sc_dmatrix_destroy (eueu_inc_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}


void 
slabs_hessian_block_mu_activation_energy_viscosity_average (slabs_hessian_params_t *hessian_params,
							    slabs_inverse_problem_params_t *inverse_params,
							    const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_activation_energy = inverse_params->grad_viscosity_activation_energy;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_activation_energy = inverse_params->grad_viscosity_second_deriv_IIe_activation_energy;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out = hessian_params->activation_energy_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *eueu_inc = hessian_params->eueu_inc;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_activation_energy_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *eueu_inc_el_mat, *grad_viscosity_second_deriv_IIe_activation_energy_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_activation_energy_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_activation_energy_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_activation_energy, grad_viscosity_activation_energy_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_activation_energy, grad_viscosity_second_deriv_IIe_activation_energy_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (eueu_inc, eueu_inc_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_activation_energy = grad_viscosity_activation_energy_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_active_energy = grad_viscosity_second_deriv_IIe_activation_energy_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict eueuinc = eueu_inc_el_mat->e[0] + nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * viscosity_IIe_deriv[0] * eueuinc[0] * grad_viscosity_activation_energy[0] / (visc[0] * visc[0])) - (average_visc_area_misfit * eueuinc[0] * viscosity_IIe_deriv[0] * grad_viscosity_activation_energy[0] / (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_activation_energy[0] * viscosity_second_deriv_IIe_active_energy[0] * eueuinc[0] / visc[0]);
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_activation_energy_el_mat);
  sc_dmatrix_destroy (grad_viscosity_activation_energy_el_mat);
  sc_dmatrix_destroy (eueu_inc_el_mat);
  sc_dmatrix_destroy (out_el_mat);

}


void 
slabs_hessian_block_mu_yield_stress_viscosity_average (slabs_hessian_params_t *hessian_params,
						       slabs_inverse_problem_params_t *inverse_params,
						       const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_yield_stress = inverse_params->grad_viscosity_yield_stress;
 ymir_vec_t           *grad_viscosity_second_deriv_IIe_yield_stress = inverse_params->grad_viscosity_second_deriv_IIe_yield_stress;
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *out = hessian_params->yield_stress_mm_average_viscosity_inc;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *eueu_inc = hessian_params->eueu_inc;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *out_el_mat;
  sc_dmatrix_t       *grad_viscosity_yield_stress_el_mat, *grad_viscosity_second_deriv_IIe_yield_stress_el_mat;;
  sc_dmatrix_t       *eueu_inc_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_yield_stress_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_yield_stress_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_yield_stress, grad_viscosity_yield_stress_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_yield_stress, grad_viscosity_second_deriv_IIe_yield_stress_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (eueu_inc, eueu_inc_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_yield_stress = grad_viscosity_yield_stress_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_yield = grad_viscosity_second_deriv_IIe_yield_stress_el_mat->e[0] + nodeid;
      double *_sc_restrict eueuinc = eueu_inc_el_mat->e[0] + nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = (average_visc_area * viscosity_IIe_deriv[0] * eueuinc[0] * grad_viscosity_yield_stress[0] / (visc[0] * visc[0])) - (average_visc_area_misfit * eueuinc[0] * viscosity_IIe_deriv[0] * grad_viscosity_yield_stress[0] / (visc[0] * visc[0])) + (average_visc_area_misfit * grad_viscosity_yield_stress[0] * viscosity_second_deriv_IIe_yield[0] * eueuinc[0] / visc[0]);
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_yield_stress_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_yield_stress_el_mat);
  sc_dmatrix_destroy (eueu_inc_el_mat);
  sc_dmatrix_destroy (out_el_mat);
}
 
void 
slabs_hessian_block_um_weakfactor_viscosity_average (slabs_hessian_params_t *hessian_params,
						     slabs_inverse_problem_params_t *inverse_params,
						     const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *u = inverse_params->u; 
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_weakfactor = inverse_params->grad_viscosity_second_deriv_IIe_weakfactor;
  ymir_vec_t           *weakzone_marker = inverse_params->weakzone_marker;
  ymir_vec_t           *out = hessian_params->weakfactor_mm_average_viscosity_inc;
  ymir_vec_t           *out1 = ymir_vec_template (viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (viscosity);
  ymir_vec_t           *out3 = ymir_vec_template (viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *u_apply1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply3 = ymir_cvec_new (mesh, 3);
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_pressure_elem_t *press_elem = inverse_params->press_elem;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *grad_viscosity_second_deriv_IIe_weakfactor_el_mat;
  sc_dmatrix_t       *grad_viscosity_weakfactor_el_mat, *weakzone_marker_el_mat;
  sc_dmatrix_t       *out1_el_mat, *out2_el_mat, *out3_el_mat; 
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_weakfactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_weakfactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  weakzone_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  out3_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (weakzone_marker, weakzone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_weakfactor, grad_viscosity_second_deriv_IIe_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_weakfactor = grad_viscosity_weakfactor_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_weak = grad_viscosity_second_deriv_IIe_weakfactor_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict output3 = out3_el_mat->e[0] +  nodeid;
 
      output1[0] = grad_viscosity_weakfactor[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);
      output2[0] = average_visc_area_misfit * viscosity_second_deriv_IIe_weak[0] / (visc[0]);
      output3[0] = -average_visc_area_misfit * grad_viscosity_weakfactor[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);


    }
     ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  ymir_stress_op_t *stress_op_mod1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod2 = ymir_stress_op_new (out2, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod3 = ymir_stress_op_new (out3, vel_dir, NULL, u, NULL);
 
  ymir_stress_op_apply (u, u_apply1, stress_op_mod1);
  ymir_stress_op_apply (u, u_apply2, stress_op_mod2);
  ymir_stress_op_apply (u, u_apply3, stress_op_mod3);
 
  ymir_vec_add (1.0, u_apply1, u_apply2);
  ymir_vec_add (1.0, u_apply2, u_apply3);

  ymir_stokes_vec_set_velocity (u_apply3,  out, press_elem);
  /* ymir_vec_copy (rhs_inc, vq_out); */
  ymir_stress_op_destroy (stress_op_mod1);
  ymir_stress_op_destroy (stress_op_mod2);
  ymir_stress_op_destroy (stress_op_mod3);
 


  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (weakzone_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_weakfactor_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_weakfactor_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);
  sc_dmatrix_destroy (out3_el_mat);
  
  ymir_vec_destroy (out1);
  ymir_vec_destroy (out2);
  ymir_vec_destroy (out3);
  ymir_vec_destroy (u_apply1);
  ymir_vec_destroy (u_apply2);
  ymir_vec_destroy (u_apply3);

  
}


void 
slabs_hessian_block_um_strain_rate_exponent_viscosity_average (slabs_hessian_params_t *hessian_params,
							       slabs_inverse_problem_params_t *inverse_params,
							       const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *u = inverse_params->u; 
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_strain_rate_exponent = inverse_params->grad_viscosity_strain_rate_exponent;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_strain_rate_exponent = inverse_params->grad_viscosity_second_deriv_IIe_strain_rate_exponent;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out = hessian_params->strain_rate_exponent_mm_average_viscosity_inc;
  ymir_vec_t           *out1 = ymir_vec_template (viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (viscosity);
  ymir_vec_t           *out3 = ymir_vec_template (viscosity);
  ymir_vec_t           *u_apply1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply3 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_pressure_elem_t *press_elem = inverse_params->press_elem;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat;
  sc_dmatrix_t       *grad_viscosity_strain_rate_exponent_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *eueu_inc_el_mat;
  sc_dmatrix_t       *out1_el_mat, *out2_el_mat, *out3_el_mat; 
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_strain_rate_exponent_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out3_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
 
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_strain_rate_exponent, grad_viscosity_strain_rate_exponent_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_strain_rate_exponent, grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);


    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_strain_rate_exponent = grad_viscosity_strain_rate_exponent_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_strain_exp = grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict output3 = out3_el_mat->e[0] +  nodeid;
 
      output1[0] = grad_viscosity_strain_rate_exponent[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);
      output2[0] = average_visc_area_misfit * viscosity_second_deriv_IIe_strain_exp[0] / (visc[0]);
      output3[0] = -average_visc_area_misfit * grad_viscosity_strain_rate_exponent[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);

    
    }

     ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_stress_op_t *stress_op_mod1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod2 = ymir_stress_op_new (out2, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod3 = ymir_stress_op_new (out3, vel_dir, NULL, u, NULL);
 
  ymir_stress_op_apply (u, u_apply1, stress_op_mod1);
  ymir_stress_op_apply (u, u_apply2, stress_op_mod2);
  ymir_stress_op_apply (u, u_apply3, stress_op_mod3);
 
  ymir_vec_add (1.0, u_apply1, u_apply2);
  ymir_vec_add (1.0, u_apply2, u_apply3);

  ymir_stokes_vec_set_velocity (u_apply3,  out, press_elem);
  /* ymir_vec_copy (rhs_inc, vq_out); */
  ymir_stress_op_destroy (stress_op_mod1);
  ymir_stress_op_destroy (stress_op_mod2);
  ymir_stress_op_destroy (stress_op_mod3);


  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_strain_rate_exponent_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);
  sc_dmatrix_destroy (out3_el_mat);

  ymir_vec_destroy (out1);
  ymir_vec_destroy (out2);
  ymir_vec_destroy (out3);
  ymir_vec_destroy (u_apply1);
  ymir_vec_destroy (u_apply2);
  ymir_vec_destroy (u_apply3);



}

void 
slabs_hessian_block_um_upper_mantle_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								 slabs_inverse_problem_params_t *inverse_params,
								 const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *u = inverse_params->u; 
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_upper_mantle = inverse_params->grad_viscosity_UM_prefactor;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_UM_prefactor = inverse_params->grad_viscosity_second_deriv_IIe_UM_prefactor;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out = hessian_params->upper_mantle_prefactor_mm_average_viscosity_inc;
  ymir_vec_t           *out1 = ymir_vec_template (viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (viscosity);
  ymir_vec_t           *out3 = ymir_vec_template (viscosity);
  ymir_vec_t           *u_apply1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply3 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_pressure_elem_t *press_elem = inverse_params->press_elem;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat;
  sc_dmatrix_t       *grad_viscosity_upper_mantle_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *out1_el_mat, *out2_el_mat, *out3_el_mat; 
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_upper_mantle_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  out3_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_upper_mantle, grad_viscosity_upper_mantle_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_UM_prefactor, grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_upper_mantle = grad_viscosity_upper_mantle_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_UM = grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict output3 = out3_el_mat->e[0] +  nodeid;
 
      output1[0] = grad_viscosity_upper_mantle[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);
      output2[0] = average_visc_area_misfit * viscosity_second_deriv_IIe_UM[0] / (visc[0]);
      output3[0] = -average_visc_area_misfit * grad_viscosity_upper_mantle[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);
 


    }
     ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  ymir_stress_op_t *stress_op_mod1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod2 = ymir_stress_op_new (out2, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod3 = ymir_stress_op_new (out3, vel_dir, NULL, u, NULL);
 
  ymir_stress_op_apply (u, u_apply1, stress_op_mod1);
  ymir_stress_op_apply (u, u_apply2, stress_op_mod2);
  ymir_stress_op_apply (u, u_apply3, stress_op_mod3);
 
  ymir_vec_add (1.0, u_apply1, u_apply2);
  ymir_vec_add (1.0, u_apply2, u_apply3);

  ymir_stokes_vec_set_velocity (u_apply3,  out, press_elem);
  /* ymir_vec_copy (rhs_inc, vq_out); */
  ymir_stress_op_destroy (stress_op_mod1);
  ymir_stress_op_destroy (stress_op_mod2);
  ymir_stress_op_destroy (stress_op_mod3);



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_upper_mantle_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);
  sc_dmatrix_destroy (out3_el_mat);
  
  ymir_vec_destroy (out1);
  ymir_vec_destroy (out2);
  ymir_vec_destroy (out3);
  ymir_vec_destroy (u_apply1);
  ymir_vec_destroy (u_apply2);
  ymir_vec_destroy (u_apply3);



}


void 
slabs_hessian_block_um_transition_zone_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								    slabs_inverse_problem_params_t *inverse_params,
								    const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *u = inverse_params->u; 
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_transition_zone = inverse_params->grad_viscosity_TZ_prefactor;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_TZ_prefactor = inverse_params->grad_viscosity_second_deriv_IIe_TZ_prefactor;
  ymir_vec_t           *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_vec_t           *out = hessian_params->transition_zone_prefactor_mm_average_viscosity_inc;
  ymir_vec_t           *out1 = ymir_vec_template (viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (viscosity);
  ymir_vec_t           *out3 = ymir_vec_template (viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *u_apply1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply3 = ymir_cvec_new (mesh, 3);
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_pressure_elem_t *press_elem = inverse_params->press_elem;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat;
  sc_dmatrix_t       *grad_viscosity_transition_zone_el_mat, *transition_zone_marker_el_mat;
  sc_dmatrix_t       *out1_el_mat, *out2_el_mat, *out3_el_mat; 
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_transition_zone_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  transition_zone_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  out3_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (transition_zone_marker, transition_zone_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_transition_zone, grad_viscosity_transition_zone_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_TZ_prefactor, grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);

    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_transition_zone = grad_viscosity_transition_zone_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_TZ = grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict output3 = out3_el_mat->e[0] +  nodeid;
 
      output1[0] = grad_viscosity_transition_zone[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);
      output2[0] = average_visc_area_misfit * viscosity_second_deriv_IIe_TZ[0] / (visc[0]);
      output3[0] = -average_visc_area_misfit * grad_viscosity_transition_zone[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);

    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_stress_op_t *stress_op_mod1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod2 = ymir_stress_op_new (out2, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod3 = ymir_stress_op_new (out3, vel_dir, NULL, u, NULL);
 
  ymir_stress_op_apply (u, u_apply1, stress_op_mod1);
  ymir_stress_op_apply (u, u_apply2, stress_op_mod2);
  ymir_stress_op_apply (u, u_apply3, stress_op_mod3);
 
  ymir_vec_add (1.0, u_apply1, u_apply2);
  ymir_vec_add (1.0, u_apply2, u_apply3);

  ymir_stokes_vec_set_velocity (u_apply3,  out, press_elem);
  /* ymir_vec_copy (rhs_inc, vq_out); */
  ymir_stress_op_destroy (stress_op_mod1);
  ymir_stress_op_destroy (stress_op_mod2);
  ymir_stress_op_destroy (stress_op_mod3);


  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (transition_zone_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_transition_zone_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);
  sc_dmatrix_destroy (out3_el_mat);
  
  ymir_vec_destroy (out1);
  ymir_vec_destroy (out2);
  ymir_vec_destroy (out3);
  ymir_vec_destroy (u_apply1);
  ymir_vec_destroy (u_apply2);
  ymir_vec_destroy (u_apply3);
}


void 
slabs_hessian_block_um_activation_energy_viscosity_average (slabs_hessian_params_t *hessian_params,
							    slabs_inverse_problem_params_t *inverse_params,
							    const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *u = inverse_params->u; 
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_activation_energy = inverse_params->grad_viscosity_activation_energy;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_activation_energy = inverse_params->grad_viscosity_second_deriv_IIe_activation_energy;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t           *out = hessian_params->activation_energy_mm_average_viscosity_inc;
  ymir_vec_t           *out1 = ymir_vec_template (viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (viscosity);
  ymir_vec_t           *out3 = ymir_vec_template (viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *u_apply1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply3 = ymir_cvec_new (mesh, 3);
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_pressure_elem_t *press_elem = inverse_params->press_elem;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *grad_viscosity_second_deriv_IIe_activation_energy_el_mat;
  sc_dmatrix_t       *grad_viscosity_activation_energy_el_mat, *upper_mantle_marker_el_mat;
  sc_dmatrix_t       *out1_el_mat, *out2_el_mat, *out3_el_mat; 
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_activation_energy_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_marker_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_activation_energy_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  out3_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_marker_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_activation_energy, grad_viscosity_activation_energy_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_activation_energy, grad_viscosity_second_deriv_IIe_activation_energy_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
   
   ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_activation_energy = grad_viscosity_activation_energy_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_active_energy = grad_viscosity_second_deriv_IIe_activation_energy_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict output3 = out3_el_mat->e[0] +  nodeid;
 
      output1[0] = grad_viscosity_activation_energy[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);
      output2[0] = average_visc_area_misfit * viscosity_second_deriv_IIe_active_energy[0] / (visc[0]);
      output3[0] = -average_visc_area_misfit * grad_viscosity_activation_energy[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);

    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }
  ymir_stress_op_t *stress_op_mod1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod2 = ymir_stress_op_new (out2, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod3 = ymir_stress_op_new (out3, vel_dir, NULL, u, NULL);
 
  ymir_stress_op_apply (u, u_apply1, stress_op_mod1);
  ymir_stress_op_apply (u, u_apply2, stress_op_mod2);
  ymir_stress_op_apply (u, u_apply3, stress_op_mod3);
 
  ymir_vec_add (1.0, u_apply1, u_apply2);
  ymir_vec_add (1.0, u_apply2, u_apply3);

  ymir_stokes_vec_set_velocity (u_apply3,  out, press_elem);
  /* ymir_vec_copy (rhs_inc, vq_out); */
  ymir_stress_op_destroy (stress_op_mod1);
  ymir_stress_op_destroy (stress_op_mod2);
  ymir_stress_op_destroy (stress_op_mod3);



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (upper_mantle_marker_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_activation_energy_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);
  sc_dmatrix_destroy (out3_el_mat);
  
  ymir_vec_destroy (out1);
  ymir_vec_destroy (out2);
  ymir_vec_destroy (out3);
  ymir_vec_destroy (u_apply1);
  ymir_vec_destroy (u_apply2);
  ymir_vec_destroy (u_apply3);


}


void 
slabs_hessian_block_um_yield_stress_viscosity_average (slabs_hessian_params_t *hessian_params,
						       slabs_inverse_problem_params_t *inverse_params,
						       const char *vtk_filepath)
{
  ymir_mesh_t          *mesh = inverse_params->mesh;
  ymir_vec_t           *u = inverse_params->u; 
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *grad_viscosity_yield_stress = inverse_params->grad_viscosity_yield_stress;
 ymir_vec_t           *grad_viscosity_second_deriv_IIe_yield_stress = inverse_params->grad_viscosity_second_deriv_IIe_yield_stress;
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *out = hessian_params->yield_stress_mm_average_viscosity_inc;
  ymir_vec_t           *out1 = ymir_vec_template (viscosity);
  ymir_vec_t           *out2 = ymir_vec_template (viscosity);
  ymir_vec_t           *out3 = ymir_vec_template (viscosity);
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *u_apply1 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply2 = ymir_cvec_new (mesh, 3);
  ymir_vec_t           *u_apply3 = ymir_cvec_new (mesh, 3);
  double               average_visc_area = inverse_params->average_visc_area;
  double               average_visc_area_misfit = inverse_params->average_visc_area_misfit;
  ymir_pressure_elem_t *press_elem = inverse_params->press_elem;
  ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_IIe_el_mat, *grad_viscosity_second_deriv_IIe_yield_stress_el_mat;
  sc_dmatrix_t       *grad_viscosity_yield_stress_el_mat;
  sc_dmatrix_t       *out1_el_mat, *out2_el_mat, *out3_el_mat; 
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_yield_stress_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_yield_stress_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out1_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out2_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  out3_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);

  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_stencil, viscosity_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_yield_stress, grad_viscosity_yield_stress_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_yield_stress, grad_viscosity_second_deriv_IIe_yield_stress_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
   
    ymir_dvec_get_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);
    ymir_dvec_get_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict grad_viscosity_yield_stress = grad_viscosity_yield_stress_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict viscosity_second_deriv_IIe_yield = grad_viscosity_second_deriv_IIe_yield_stress_el_mat->e[0] + nodeid;
      double *_sc_restrict output1 = out1_el_mat->e[0] +  nodeid;
      double *_sc_restrict output2 = out2_el_mat->e[0] +  nodeid;
      double *_sc_restrict output3 = out3_el_mat->e[0] +  nodeid;
 
      output1[0] = grad_viscosity_yield_stress[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);
      output2[0] = average_visc_area_misfit * viscosity_second_deriv_IIe_yield[0] / (visc[0]);
      output3[0] = -average_visc_area_misfit * grad_viscosity_yield_stress[0] * viscosity_IIe_deriv[0] / (visc[0] * visc[0]);


    }
    ymir_dvec_set_elem (out1, out1_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out2, out2_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
     ymir_dvec_set_elem (out3, out3_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

 
  }
  ymir_stress_op_t *stress_op_mod1 = ymir_stress_op_new (out1, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod2 = ymir_stress_op_new (out2, vel_dir, NULL, u, NULL);
  ymir_stress_op_t *stress_op_mod3 = ymir_stress_op_new (out3, vel_dir, NULL, u, NULL);
 
  ymir_stress_op_apply (u, u_apply1, stress_op_mod1);
  ymir_stress_op_apply (u, u_apply2, stress_op_mod2);
  ymir_stress_op_apply (u, u_apply3, stress_op_mod3);
 
  ymir_vec_add (1.0, u_apply1, u_apply2);
  ymir_vec_add (1.0, u_apply2, u_apply3);

  ymir_stokes_vec_set_velocity (u_apply3,  out, press_elem);
  /* ymir_vec_copy (rhs_inc, vq_out); */
  ymir_stress_op_destroy (stress_op_mod1);
  ymir_stress_op_destroy (stress_op_mod2);
  ymir_stress_op_destroy (stress_op_mod3);

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  inverse_viscosity, "inverse viscosity", */
  /* 		  NULL); */



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (grad_viscosity_yield_stress_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_yield_stress_el_mat);
  sc_dmatrix_destroy (out1_el_mat);
  sc_dmatrix_destroy (out2_el_mat);
  sc_dmatrix_destroy (out3_el_mat);
  
  ymir_vec_destroy (out1);
  ymir_vec_destroy (out2);
  ymir_vec_destroy (out3);
  ymir_vec_destroy (u_apply1);
  ymir_vec_destroy (u_apply2);
  ymir_vec_destroy (u_apply3);
 
}


void
solve_dense_hessian (slabs_inverse_problem_params_t *inverse_params,
		     slabs_hessian_params_t *hessian_params)

{
  ymir_mesh_t *mesh = inverse_params->mesh;
  MPI_Comm mpicomm = mesh->ma->mpicomm;
#ifdef YMIR_PETSC
  Mat            A;
  KSP            solver;
  PC             pc;
  PetscMPIInt    rank;
  PetscReal      norm,v,v1,v2,maxval;
  PetscInt       i, j,  n = 3,maxind, nlocal, size=3;
  Vec            X1, Y1;
  double         pi = 3.14;
  PetscErrorCode ierr;
  PetscScalar    one = pi,two = 2.0,three = 3.0,dots[3],dot, *array, *a, *b, *c;
  double new = one;

#endif  
  
  ierr = PetscMalloc(size*size*sizeof(PetscScalar),&a);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscScalar),&b);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscScalar),&c);CHKERRQ(ierr);
    
  for (i=0; i<size; i++) {
    b[i] = i;
    for (j=0; j<size; j++) {
      a[i+j*size] =  rand() / 1.e8;
    }
  }

  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,1,size,b,&X1);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X1);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X1);CHKERRQ(ierr);
  ierr = VecView(X1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecCreateSeqWithArray(MPI_COMM_SELF,1,size,c,&Y1);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Y1);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y1);CHKERRQ(ierr);


  ierr = MatCreate(MPI_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,size,size,size,size);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(A,a);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);


  ierr = KSPCreate(mpicomm, &solver);CHKERRQ(ierr);
  ierr = KSPSetType(solver,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(solver,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = KSPSetOperators(solver,A,A);CHKERRQ(ierr);
  ierr = KSPSolve(solver,X1,Y1);CHKERRQ(ierr);
  ierr = VecView(X1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(Y1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = KSPDestroy(&solver);CHKERRQ(ierr);

  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  ierr = PetscFree(c); CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&X1);CHKERRQ(ierr);
  ierr = VecDestroy(&Y1);CHKERRQ(ierr);


}



#if 0

static void
slabs_solve_optimization  (slabs_inverse_problem_params_t *inverse_params,
			   slabs_physics_options_t *physics_options,
			   slabs_discr_options_t *discr_options,
			   slabs_nl_solver_options_t *solver_options)
{
    double                cost;
    ymir_mesh_t           *mesh = inverse_params->mesh;
    ymir_pressure_elem_t  *press_elem = inverse_params->stokes_pc->stokes_op->press_elem;
    ymir_vec_t            *uobs = inverse_params->uobs;
    ymir_vec_t            *adjoint_vq = inverse_params->adjoint_vq;
    ymir_vec_t            *old_up = ymir_vec_template (adjoint_vq);
    ymir_dvec_t           *old_grad = ymir_vec_template (state->weak_vec);
    ymir_dvec_t           *grad = ymir_dvec_new (mesh, 2, YMIR_GAUSS_NODE);
    ymir_dvec_t           *grad_weak_factor = ymir_vec_template (state->weak_vec);
    ymir_dvec_t           *grad_strain_exp  = ymir_vec_template (state->weak_vec);
    ymir_dvec_t           *grad_yield_stress = ymir_vec_template (state->weak_vec);
    ymir_dvec_t           *weak_zone_stencil = inverse_params->weak_zone_stencil;
    ymir_dvec_t           *upper_mantle_stencil = inverse_params->upper_mantle_stencil;
    ymir_dvec_t           *weak_factor = ymir_vec_template (state->weak_vec);
    ymir_dvec_t           *inc_update = ymir_dvec_new (mesh, 2, YMIR_GAUSS_NODE);
    ymir_dvec_t           *ones = ymir_vec_template (state->weak_vec);
    ymir_dvec_t           *upper_mantle = ymir_vec_template (state->weak_vec);
    ymir_dvec_t           *grad_all = ymir_vec_template (grad);
    ymir_dvec_t           *proj_mass = ymir_vec_template (state->weak_vec);
    double                gradnorm, gradnormprev, orignorm = 0;
    double                grad_proj[3], hessian[9], update[3];
    double                proj_grad_prefactor, proj_grad_strain_exp;
    double                proj_grad_yield_stress;
    double                state_rtol;
    double                adjoint_rtol;
    double                inc_rtol;
    double                alpha;
    double                armijo_c1 = 1.0e-4;
    double                expa = 1.6180339887;
    int                   i = 0;
    double                old_prefactor;
    double                old_strain_exp;
    double                old_yield_stress;
    double                new_cost;
    slabs_nl_stokes_problem_t *nl_stokes_resolve;

    /* For the optimization problem we will first compute the adjoint solution given
       the surface observations and guessed solution.
       1) Guessed solution is stored in state->up
       2) Observation is stored in inverse_params->uobs
       3) Parameters(rheological) at current interation are in inverse_params->(n, prefactor, etc).
    */


    /* Initialize arrays for gradient, hessian, and update. Need to find a better way to do this */
    
    grad_proj[0] = 0;
    grad_proj[1] = 0;
    grad_proj[2] = 0;

    hessian[0] = 0;
    hessian[1] = 0;
    hessian[2] = 0;
    hessian[3] = 0;
    hessian[4] = 0;
    hessian[5] = 0;
    hessian[6] = 0;
    hessian[7] = 0;
    hessian[8] = 0;

    
    update[0] = 0;
    update[1] = 0;
    update[2] = 0;


    /* Compute adjoint solution where the RHS is the misfit in surface
       velocities.
    */
    
    
    YMIR_GLOBAL_PRODUCTION ("Entering Optimization Routine.\n");
    ymir_vec_set_zero (adjoint_vq);
    YMIR_GLOBAL_PRODUCTION ("Set adjoint rhs to zero.\n");
    
    
    slabs_nonlinear_solver_solve_adjoint (adjoint_vq, state,
					  uobs, lin_stokes,
					  &nl_stokes, press_elem,
					  physics_options,
					  discr_options,
					  solver_options, NULL,
					  NULL);


    /* Compute cost functional */
    cost = slabs_surface_vel_misfit (uobs, state, press_elem);
    YMIR_GLOBAL_PRODUCTIONF ("Optimization cost functional %E\n", cost);
    
    
    /* Compute the gradient: need to make option for different parameter inversion
     */

    slabs_compute_gradient_weakfactor (grad_weak_factor, state, adjoint_vq,
    				       physics_options, lin_stokes, &nl_stokes,
    				       mesh, press_elem, NULL);
    
    slabs_compute_gradient_strain_rate_exponent (grad_strain_exp, state, adjoint_vq,
    						 physics_options, lin_stokes, &nl_stokes,
    						 mesh, press_elem, NULL);

    slabs_compute_gradient_yield_stress (grad_yield_stress, state, adjoint_vq,
    					 physics_options, lin_stokes, &nl_stokes,
    					 mesh, press_elem, NULL);
    /* Need to project gradient */
    proj_grad_prefactor = ymir_vec_innerprod (grad_weak_factor, weak_zone_stencil);
    proj_grad_strain_exp = ymir_vec_innerprod (grad_strain_exp, upper_mantle_stencil);
    proj_grad_yield_stress = ymir_vec_innerprod (grad_yield_stress, ones);


    grad_proj[0] = -proj_grad_prefactor;
    grad_proj[1] = -proj_grad_strain_exp;
    grad_proj[2] = -proj_grad_yield_stress;

    YMIR_GLOBAL_PRODUCTIONF ("gradient of prefactor: %g.\n", grad_proj[0]);
    YMIR_GLOBAL_PRODUCTIONF ("gradient of strain exp: %g.\n", grad_proj[1]);
    YMIR_GLOBAL_PRODUCTIONF ("gradient of yield stress: %g.\n", grad_proj[2]);
	

    
    /* Compute initial gradient norm */
    inv_grad_mass = ymir_vec_template (grad);
    gradmass = ymir_vec_template (grad);
    ymir_mass_lump (inv_grad_mass);
    ymir_vec_reciprocal (inv_grad_mass);
    ymir_vec_multiply (grad, inv_grad_mass, gradmass);
    gradnorm = sqrt (ymir_vec_innerprod (gradmass, grad));
    
    /* stats [11] = gradnorm; */

    
    double rel_tol = 1.0e-3;
    double max_iter = 20;
    double max_line_search = 40;
    double state_rtol_max = 1.0e-6;
    double state_rtol_min = 1.0e-6;
    double adjoint_rtol_min = 1.0e-6;
    double adjoint_rtol_max = 1.0e-6;
    double inc_rtol_min = 1.0e-6;
    double inc_rtol_max = 1.0e-6;
    double factor;
    double old_cg_rtol;
    
    /* Note this is the weak factor stencil: computed only once */
    slabs_physics_compute_weak_factor_distrib (weak_factor, physics_options);
    slabs_physics_compute_UM_distrib (upper_mantle, physics_options);
  
    YMIR_GLOBAL_VERBOSEF ("GRadient norm: %E\n", gradnorm);
    
    
    for (i = 0; i < 25; i++) {
      
      if (!i) {
	orignorm = gradnorm;
      }
#if 0
      /* if the norm of the gradient is below the threshold */
      if (gradnorm < orignorm * rel_tol) {
	break;
      }
#endif
      /* if the maximum amount of iterations has be reached, exit */
      if (i >= max_iter) {
	break;
      }
      
      ymir_vec_set_zero (inc_update);
      ymir_vec_set_zero (dprefactor);
      ymir_vec_set_zero (dstrain_exp);
      ymir_vec_set_zero (dyield_stress);

      /* if (i > 2){ */
      /* 	cg_rtol = 1.0e-2; */
      /* } */

      YMIR_GLOBAL_STATISTICSF ("Optimization iter %d, rtol %E\n", i, cg_rtol);
      /* Solve KKT system for descent direction */
      /* slabs_cg_gauss_newton_hessian (inc_update, grad, cg_rtol, inverse_params); */
      /* Explicitly form the Hessian */
#if 1
      slabs_hessian3d_construct_explicit (mesh, &hessian, inverse_params);
      slabs_solve_hessian3d_direct (&update, &hessian, &grad_proj);
#endif
      /* If outer iteration > 1 then destroy previous adjoint/Newton operators and
	 preconditioners */
      if (i > 0){
      slabs_nonlinear_stokes_op_pc_destroy (nl_stokes_resolve);
      slabs_nonlinear_stokes_problem_destroy (nl_stokes_resolve);
      }
 

      /* searchdotdescent is the inner product of the update from the Hessian and gradient */
      double search_dot_descent;
      /* search_dot_descent = ymir_vec_innerprod (dprefactor, grad); */
      /* YMIR_GLOBAL_VERBOSEF ("search/descent inner product %E\n", search_dot_descent); */
      
#if 0
      if (searchdotdescent <= 0.) {
	/* This shouldn't happen for G-N but, can be useful and should be implemented
	   for full Hessian */
	YMIR_GLOBAL_LERROR ("Warning: search direction is not descent direction\n");
      }
#endif
      
      
      /* Need to save old information (velocity, pressure, etc) */
      old_strain_exp = physics_options->viscosity_stress_exponent;
      old_prefactor = physics_options->toy_weakzone_plate1;
      old_yield_stress = physics_options->viscosity_stress_yield;
      old_activation_energy = physics_options->viscosity_temp_decay;
      alpha = 1.0;
      
      /* add the scaled update to the old parameters. First we must project the descent
	 direction to a single prefactor. We do this by taking the inner product of the stencil
	 of the weak zone and and descent direction.
      */
      double proj_prefactor;
      double proj_strain_exp;
      double proj_yield_stress;
      double proj_gradient_prefactor;
      double proj_gradient_strain_exp;
      double proj_gradient_yield_stress;


      int line_search_count = 0;
 
      /* Projected values */
      proj_gradient_prefactor =  -grad_proj[0];
      proj_gradient_strain_exp = -grad_proj[1];
      proj_gradient_yield_stress = -grad_proj[2];


      YMIR_GLOBAL_PRODUCTIONF ("Update for the prefactor is: %g.\n", update[0]);
      YMIR_GLOBAL_PRODUCTIONF ("Update for the strain rate exponent is: %g.\n", update[1]);
      YMIR_GLOBAL_PRODUCTIONF ("Update for the yield stress is: %g.\n", update[2]);


      do {
	double prefactor;
	double strain_exp;
	double yield_stress;
	double rescale = 1.0;
        double    prefactor_scale = 1.0;
	double    strain_exp_scale = 1.0;


	prefactor = log (old_prefactor);
	prefactor += lsalpha * prefactor_scale   * update[0];
	/* need to transform the 'log prefactor' to exp(prefactor)' */
	prefactor = exp (prefactor);
	YMIR_GLOBAL_PRODUCTIONF ("old prefactor is: %g, New prefactor is: %g.\n", old_prefactor, prefactor);

	strain_exp = log (old_strain_exp);
	strain_exp += lsalpha * strain_exp_scale * update[1];
	/* need to transform the 'log prefactor' to exp(prefactor)' */
	strain_exp = exp (strain_exp);
	YMIR_GLOBAL_PRODUCTIONF ("old strain exp is: %g, New strain rate exponent is: %g.\n",
				 old_strain_exp, strain_exp);


	yield_stress = log (old_yield_stress);
	yield_stress += lsalpha * update[2];
	/* need to transform the 'log prefactor' to exp(prefactor)' */
	yield_stress = exp (yield_stress);
	YMIR_GLOBAL_PRODUCTIONF ("old yield stress is: %g, New yield stress is: %g.\n",
				 old_yield_stress, yield_stress);


	/* recompute weak zone. First we set the velocity-pressure vec to zero and change
	   the vale of the prefactor.
	*/
       	double prefactor_new = prefactor;
	double strain_exp_new = strain_exp;
	double yield_stress_new = yield_stress;
	ymir_vec_set_zero (state->vel_press_vec);
	slabs_update_prefactor_simple (prefactor_new, physics_options);
 	slabs_update_strain_exp (strain_exp_new, physics_options);
	slabs_update_yield_stress (yield_stress_new, physics_options);
	slabs_physics_compute_weakzone (state->weak_vec, physics_options);
  
	/* Create new stokes problem if we need to backtrack */
     
	nl_stokes_resolve = slabs_nonlinear_stokes_problem_new (state, mesh, press_elem,
								physics_options);
	
	slabs_solve_stokes (lin_stokes, &nl_stokes_resolve, p8est, &mesh, &press_elem,
			    state, physics_options, discr_options,
			    solver_options, NULL, NULL, NULL);
  
	/* Now compute new misfit in surface velocities */
	new_cost = slabs_surface_vel_misfit (uobs, state, press_elem);

	YMIR_GLOBAL_VERBOSEF ("line search: alpha %f old cost %E new cost %E\n",
			      lsalpha, cost, new_cost);
      

	/* Use globalization via Armijo Line search: If (J(m_k+1)<J(m_k)+c*alpha*d*g^T), we
	   accept the new solution (as well as the operator used to compute next adjoint
	   solution). */

	  if (new_cost < (cost - lsalpha * armijo_c1 * ( proj_gradient_prefactor * update[0] +
							  proj_gradient_strain_exp * update[1] +
							 proj_gradient_yield_stress * update[2]))){


	  YMIR_GLOBAL_PRODUCTIONF ("Cost function minimized in %i line searches for G-N iteration %i.\n",
				   line_search_count, i);

	  /* Since the misfit is minimized from this new parameter, we save the solution
	     (u,p) along with the adjoint operator, etc. Since the nl-stokes op is destroyed, we will
	     recompute it here.
	  */
	
	  /* Recompute the adjoint operator since it was destroyed. */
	  slabs_nonlinear_stokes_op_new (nl_stokes_resolve, state, physics_options, 0,
					 solver_options->nl_solver_type,
					 solver_options->nl_solver_primaldual_type,
					 solver_options->nl_solver_primaldual_scal_type, NULL);
  
	  slabs_nonlinear_stokes_pc_new (nl_stokes_resolve, state,
					 solver_options->schur_diag_type,
					 solver_options->scaling_type, 0, NULL);

	  
	  
	  inverse_params->viscosity = nl_stokes_resolve->viscosity;
 	  inverse_params->prefactor = prefactor;
	  inverse_params->strain_exp = strain_exp;
	  inverse_params->yield_stress = yield_stress;
	  inverse_params->stokes_pc = nl_stokes_resolve->stokes_pc;
	  inverse_params->yielding_marker = nl_stokes_resolve->yielding_marker;
	  inverse_params->bounds_marker = nl_stokes_resolve->bounds_marker;

	  YMIR_GLOBAL_PRODUCTIONF (" The new prefactor is: %g.\n", inverse_params->prefactor);
	  YMIR_GLOBAL_PRODUCTIONF (" The new strain rate exponent is: %g.\n", inverse_params->strain_exp);
	  YMIR_GLOBAL_PRODUCTIONF (" The new yield stress is: %g.\n", inverse_params->yield_stress);

	  break;

	}
	else {
	  slabs_nonlinear_stokes_problem_destroy (nl_stokes_resolve);
	  lsalpha /= 2.0;
	  line_search_count++;
	}
      
      } while ((max_line_search > line_search_count));
      
      /* We assign the new cost that needs to be minimized */
      cost = new_cost;
      gradnormprev = gradnorm;
      
      /* Compute the new gradient norm */
      ymir_vec_multiply (grad, inv_grad_mass, gradmass);
      gradnorm = sqrt (ymir_vec_innerprod (gradmass, grad));


      YMIR_GLOBAL_PRODUCTIONF ("New compute optimization cost functional %E\n", cost);
      YMIR_GLOBAL_PRODUCTIONF ("Entering Gauss-Newton iteration %d\n.", i+1);

      /* Compute the adjoint solution from the new surface velocities */
      ymir_vec_set_zero (adjoint_vq);
      slabs_nonlinear_solver_solve_adjoint (adjoint_vq, state,
					    uobs, lin_stokes,
					    &nl_stokes_resolve, press_elem,
					    physics_options,
					    discr_options,
					    solver_options, NULL,
					    NULL);

      /* Now compute the gradient to be used as the RHS for Hessian system */
      ymir_vec_set_zero (grad);
      slabs_compute_gradient_weakfactor (grad_weak_factor, state, adjoint_vq,
      					 physics_options, lin_stokes,
      					 &nl_stokes_resolve,
      					 mesh, press_elem, NULL);
 

      slabs_compute_gradient_strain_rate_exponent (grad_strain_exp, state, adjoint_vq,
						   physics_options, lin_stokes,
						   &nl_stokes_resolve,
      						   mesh, press_elem, NULL);
 

      slabs_compute_gradient_yield_stress (grad_yield_stress, state, adjoint_vq,
      					   physics_options, lin_stokes,
      					   &nl_stokes_resolve,
      					   mesh, press_elem, NULL);
 


      /* Need to project gradient */
    proj_grad_prefactor = ymir_vec_innerprod (grad_weak_factor, weak_zone_stencil);
    proj_grad_strain_exp = ymir_vec_innerprod (grad_strain_exp, upper_mantle_stencil);
    proj_grad_yield_stress = ymir_vec_innerprod (grad_yield_stress, ones);
    
    grad_proj[0] = -proj_grad_prefactor;
    grad_proj[1] = -proj_grad_strain_exp;
    grad_proj[2] = -proj_grad_yield_stress;


    YMIR_GLOBAL_PRODUCTIONF ("gradient of prefactor: %g.\n", grad_proj[0]);
    YMIR_GLOBAL_PRODUCTIONF ("gradient of strain exp: %g.\n", grad_proj[1]);
    YMIR_GLOBAL_PRODUCTIONF ("gradient of strain yield stress: %g.\n", grad_proj[2]);
	

    
}

    /* Final destruction of adjoint/Newton operators and nl-stokes problem */
    slabs_nonlinear_stokes_op_pc_destroy (nl_stokes_resolve);
    slabs_nonlinear_stokes_problem_destroy (nl_stokes_resolve);

  
    ymir_vec_destroy (old_up);
    ymir_vec_destroy (old_grad);
    ymir_vec_destroy (grad);
    ymir_vec_destroy (grad_all);
    ymir_vec_destroy (grad_weak_factor);
    ymir_vec_destroy (grad_strain_exp);
    ymir_vec_destroy (grad_yield_stress);
    ymir_vec_destroy (gradmass);
    ymir_vec_destroy (inv_grad_mass);
    ymir_vec_destroy (weak_factor);
    ymir_vec_destroy (upper_mantle);
    ymir_vec_destroy (ones);
    ymir_vec_destroy (inc_update);
    ymir_vec_destroy (dprefactor);
    ymir_vec_destroy (dstrain_exp);
    ymir_vec_destroy (dyield_stress);
    ymir_vec_destroy (proj_mass);
    
}


#endif





#if 0
void
slabs_viscosity_IIe_second_deriv (slabs_inverse_problem_params_t *inverse_params,
				  const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *IIe = inverse_params->IIe;
  ymir_vec_t           *yielding_marker = inverse_params->yielding_marker;
  ymir_vec_t           *bounds_marker = 
  ymir_vec_t           *grad_viscosity_transition_zone = inverse_params->grad_viscosity_transition_zone;
  ymir_vec_t           *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat;
  sc_dmatrix_t       *transition_zone_stencil_el_mat;
  double             *x, *y, *z, *tmp_el;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  transition_zone_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
  x = YMIR_ALLOC (double, n_nodes_per_el);
  y = YMIR_ALLOC (double, n_nodes_per_el);
  z = YMIR_ALLOC (double, n_nodes_per_el);
  tmp_el = YMIR_ALLOC (double, n_nodes_per_el);
     

 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    slabs_elem_get_gauss_coordinates (x, y, z, elid, mangll, tmp_el);
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (transition_zone_marker, transition_zone_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_transition_zone, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_deriv = grad_viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict transition_zone = transition_zone_stencil_el_mat->e[0] +  nodeid;
 
      viscosity_deriv[0] = transition_zone[0] * visc[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_transition_zone, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
  }

  snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  grad_viscosity_transition_zone, "gradient of viscosity w.r.t transition_zone",
		  NULL);

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (transition_zone_stencil_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);
  YMIR_FREE (x);
  YMIR_FREE (y);
  YMIR_FREE (z);
  YMIR_FREE (tmp_el);



}


#endif


/* static void */
/* slabs_hessian_block_mm_prefactor (ymir_vec_t *gamma_inc, */
/* 				  slabs_inverse_problem_params_t *inverse_params) */

/* { */
/*   ymir_mesh_t *mesh = gamma_inc->mesh; */
/*   ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc; */
/*   ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem; */
/*   ymir_dvec_t *viscosity = inverse_params->viscosity; */
/*   ymir_vec_t *adjoint_vq = inverse_params->adjoint_vq; */
/*   ymir_cvec_t *usol = stokes_pc->stokes_op->stress_op->usol; */
/*   ymir_cvec_t *v = ymir_cvec_new (mesh, 3); */
/*   ymir_dvec_t *eu = ymir_dvec_new (mesh, 6, YMIR_GAUSS_NODE); */
/*   ymir_dvec_t *ev = ymir_dvec_new (mesh, 6, YMIR_GAUSS_NODE); */
/*   ymir_dvec_t *euev = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE); */
/*   ymir_dvec_t *visc_temp = ymir_vec_clone (viscosity); */
/*   ymir_dvec_t *ones = ymir_vec_template (viscosity); */
/*   ymir_dvec_t *no_yielding = ymir_vec_template (viscosity); */
/*   ymir_dvec_t *mass_gamma = ymir_vec_template (gamma_inc); */

/*   /\* compute strain rate tensors e(u) and e(v) *\/ */
/*   ymir_velocity_strain_rate (usol, eu, 0); */
/*   ymir_stokes_vec_get_velocity (adjoint_vq, v, press_elem); */
/*   ymir_velocity_strain_rate (v, ev, 0); */
/*   ymir_vec_destroy (v); */

/*   /\* compute e(u):e(v) *\/ */
/*   ymir_velocity_symtens_dotprod (eu, ev, euev); */
/*   YMIR_GLOBAL_PRODUCTION ("Computed e(u):e(v).\n"); */
/*   ymir_vec_destroy (eu); */
/*   ymir_vec_destroy (ev); */
  
/*   #if 1 */
/*   /\* set the values for the non-yielding vector *\/ */
/*   ymir_vec_set_value (no_yielding, 1.0); */
/*   ymir_vec_add (-1.0, inverse_params->yielding_marker, no_yielding); */
/*   #endif */


/*   /\* need to adjust the viscosity to remove eta_min *\/ */
/*   ymir_vec_set_value (ones, 1.0); */
/*   ymir_vec_add (-2.0e-2, ones, visc_temp); */
/*   ymir_vec_destroy (ones); */
/*   ymir_dvec_multiply_in1 (no_yielding, visc_temp); */

/*   /\* start forming block now *\/ */
/*   ymir_vec_multiply_in (visc_temp, euev); */
/*   ymir_vec_destroy (visc_temp); */
/*   ymir_mass_apply (euev, gamma_inc); */
 
/*   ymir_vec_destroy (euev); */
/*   ymir_vec_destroy (no_yielding); */

/* } */

/* static void */
/* slabs_hessian_block_mm_yield_stress (ymir_dvec_t *yield_inc, */
/* 				     slabs_inverse_problem_params_t *inverse_params) */

/* { */
/*   ymir_mesh_t *mesh = yield_inc->mesh; */
/*   ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc; */
/*   ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem; */
/*   ymir_dvec_t *viscosity = inverse_params->viscosity; */
/*   ymir_vec_t *adjoint_vq = inverse_params->adjoint_vq; */
/*   ymir_cvec_t *usol = stokes_pc->stokes_op->stress_op->usol; */
/*   ymir_cvec_t *v = ymir_cvec_new (mesh, 3); */
/*   ymir_dvec_t *eu = ymir_dvec_new (mesh, 6, YMIR_GAUSS_NODE); */
/*   ymir_dvec_t *ev = ymir_dvec_new (mesh, 6, YMIR_GAUSS_NODE); */
/*   ymir_dvec_t *euev = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE); */
/*   ymir_dvec_t *visc_temp = ymir_vec_clone (viscosity); */
/*   ymir_dvec_t *ones = ymir_vec_template (viscosity); */
/*   ymir_dvec_t *mass_gamma = ymir_vec_template (yield_inc); */

/*   /\* compute strain rate tensors e(u) and e(v) *\/ */
/*   ymir_velocity_strain_rate (usol, eu, 0); */
/*   ymir_stokes_vec_get_velocity (adjoint_vq, v, press_elem); */
/*   ymir_velocity_strain_rate (v, ev, 0); */
/*   ymir_vec_destroy (v); */

/*   /\* compute e(u):e(v) *\/ */
/*   ymir_velocity_symtens_dotprod (eu, ev, euev); */
/*   YMIR_GLOBAL_PRODUCTION ("Computed e(u):e(v).\n"); */
/*   ymir_vec_destroy (eu); */
/*   ymir_vec_destroy (ev); */
  


/*   /\* need to adjust the viscosity to remove eta_min *\/ */
/*   ymir_vec_set_value (ones, 1.0); */
/*   ymir_vec_add (-2.0e-2, ones, visc_temp); */
/*   ymir_vec_destroy (ones); */
/*   ymir_dvec_multiply_in1 (inverse_params->yielding_marker, visc_temp); */

/*   /\* start forming block now *\/ */
/*   ymir_vec_multiply_in (visc_temp, euev); */
/*   ymir_vec_destroy (visc_temp); */
/*   ymir_mass_apply (euev, yield_inc); */
 
/*   ymir_vec_destroy (euev); */

/* } */


/* static void  */
/* slabs_hessian_block_mm_strain_exp (ymir_vec_t *strain_exp_out, */
/* 				   slabs_inverse_problem_params_t *inverse_params) */

/* { */

/*   ymir_mesh_t *mesh = strain_exp_out->mesh; */
/*   ymir_stokes_pc_t *stokes_pc = inverse_params->stokes_pc; */
/*   ymir_pressure_elem_t *press_elem = stokes_pc->stokes_op->press_elem; */
/*   ymir_dvec_t *eu = ymir_dvec_new (mesh, 6, YMIR_GAUSS_NODE); */
/*   ymir_dvec_t *ev = ymir_dvec_new (mesh, 6, YMIR_GAUSS_NODE); */
/*   ymir_dvec_t *eu_ev_innerprod = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);  */
/*   ymir_dvec_t *IIe = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE); */
/*   ymir_cvec_t *u = stokes_pc->stokes_op->stress_op->usol; */
/*   ymir_cvec_t *v = ymir_cvec_new (mesh, 3); */
/*   ymir_vec_t  *vq = inverse_params->adjoint_vq; */
/*   ymir_dvec_t *ones = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE); */
/*   ymir_dvec_t *viscosity = inverse_params->viscosity; */
/*   ymir_vec_t  *weak_vec = inverse_params->weak_zone_stencil; */
/*   ymir_dvec_t *no_yielding = ymir_vec_template (viscosity); */
/*   ymir_dvec_t *strain_exp_out2 = ymir_vec_template (viscosity); */
/*   ymir_dvec_t *visc_temp = ymir_vec_clone (viscosity); */
/*   ymir_dvec_t *visc_temp2 = ymir_vec_clone (viscosity); */
/*   ymir_dvec_t *strain_exp = ymir_vec_template (strain_exp_out); */
/*   ymir_velocity_elem_t *vel_elem; */
/*   double n = inverse_params->strain_exp; */
/*   double scale = -0.5 / (n); */

/*   YMIR_GLOBAL_PRODUCTION ("Entering hessian block mv.\n"); */
/*   /\* Compute e(u) strain rate tensor *\/ */
/*   ymir_velocity_strain_rate (u, eu, 0); */
/*   YMIR_GLOBAL_PRODUCTION ("constructed strain rate tensor.\n"); */


/*   /\* Compute e(v_inc) strain rate tensor *\/ */
/*   ymir_stokes_vec_get_velocity (vq, v, press_elem); */
/*   ymir_velocity_strain_rate (v, ev, 0); */
/*   YMIR_GLOBAL_PRODUCTION ("constructed incremental strain rate tensor.\n"); */


/*   /\* compute the inner-product of strain rate tensors e(u):e(v) *\/ */
/*   ymir_velocity_symtens_dotprod (eu, ev, eu_ev_innerprod); */
/*   YMIR_GLOBAL_PRODUCTION ("Constructed eu:ev.\n"); */
 
/*   #if 1 */
/*   /\* set the values for the non-yielding vector *\/ */
/*   ymir_vec_set_value (no_yielding, 1.0); */
/*   ymir_vec_add (-1.0, inverse_params->yielding_marker, no_yielding); */
/*   #endif */


/*   /\* need to multiply by ln(second invariant ^ power) *. First take second invariant and raise */
/*      to -1/2n^2 *\/ */
/*   vel_elem = ymir_velocity_elem_new (mesh->cnodes->N); */
/*   ymir_second_invariant_vec (u, IIe, vel_elem); */

/*   /\* Going to compute the log of the second invariant *\/ */
/*   sc_dmatrix_t       *view = sc_dmatrix_new (0, 0); */
/*   sc_dmatrix_t       *IIeview = sc_dmatrix_new (0, 0); */
/*   ymir_dvec_t        *IIe_clone = ymir_vec_clone (IIe); */
/*   ymir_dvec_t        *log_IIe = ymir_vec_template (IIe); */
/*   int                 i; */
 
/*   ymir_vec_set_zero (log_IIe); */
/*   ymir_dvec_get_all (IIe_clone, view, YMIR_DVEC_STRIDE, YMIR_READ); */
/*   ymir_dvec_get_all (log_IIe, IIeview, YMIR_DVEC_STRIDE, YMIR_WRITE); */

/*   for (i = 0; i < view->m * view->n; i++) { */
/*     IIeview->e[0][i] = log (view->e[0][i]); */
/*   } */

/*   ymir_read_view_release (view); */
/*   ymir_dvec_set_all (log_IIe, IIeview, YMIR_DVEC_STRIDE, YMIR_SET); */

/*   sc_dmatrix_destroy (view); */
/*   sc_dmatrix_destroy (IIeview); */
/*   ymir_vec_destroy (IIe_clone); */



/*   /\* compute viscosity component part *\/ */
/*   ymir_vec_set_value (ones, 1.0); */
/*   ymir_vec_add (-2.0e-2, ones, visc_temp); */
/*   ymir_dvec_multiply_in1 (no_yielding, visc_temp); */
/*   ymir_vec_multiply_in1 (weak_vec, visc_temp); */
/*   ymir_vec_multiply_in1 (log_IIe, visc_temp); */
/*   ymir_vec_scale (scale, visc_temp); */
/*   ymir_vec_multiply_in1 (eu_ev_innerprod, visc_temp); */


/*   /\* second part *\/ */
/*   ymir_vec_set_value (ones, 1.0); */
/*   ymir_vec_add (-2.0e-2, ones, visc_temp2); */
/*   ymir_dvec_multiply_in1 (no_yielding, visc_temp2); */
/*   ymir_vec_multiply_in1 (weak_vec, visc_temp2); */
/*   ymir_vec_multiply_in1 (log_IIe, visc_temp2); */
/*   ymir_vec_multiply_in1 (log_IIe, visc_temp2); */
/*   ymir_vec_scale (scale, visc_temp2); */
/*   ymir_vec_scale (scale, visc_temp2); */
/*   ymir_vec_multiply_in1 (eu_ev_innerprod, visc_temp2); */


/*   ymir_vec_destroy (ones); */
/*   YMIR_GLOBAL_PRODUCTION ("modify viscosity.\n"); */






/*   /\* Now need to add in the second term *\/ */
/*   ymir_vec_add (1.0, visc_temp, visc_temp2); */
/*   ymir_mass_apply (visc_temp2, strain_exp_out); */

/*   ymir_vec_destroy (eu); */
/*   ymir_vec_destroy (IIe); */
/*   ymir_vec_destroy (log_IIe); */
/*   ymir_vec_destroy (eu_ev_innerprod); */
/*   ymir_vec_destroy (ev); */
/*   ymir_vec_destroy (v); */
/*   ymir_vec_destroy (visc_temp); */
/*   ymir_vec_destroy (visc_temp2); */
/*   ymir_vec_destroy (no_yielding); */
/*   ymir_velocity_elem_destroy (vel_elem); */
/*   YMIR_GLOBAL_PRODUCTION ("Leaving hessian block_mv_strain_exp.\n"); */
  
/* } */




 



/* static void  */
/* slabs_solve_optimization  (slabs_inverse_problem_params_t *inverse_params, */
/* 			   slabs_stokes_state_t *state,			    */
/* 			   slabs_lin_stokes_problem_t *lin_stokes, */
/* 			   slabs_nl_stokes_problem_t *nl_stokes, */
/* 			   p8est_t *p8est, */
/* 			   slabs_inverse_problem_params_t *inverse_params, */
/* 			   slabs_physics_options_t *physics_options, */
/* 			   slabs_discr_options_t *discr_options, */
/* 			   slabs_nl_solver_options_t *solver_options) */
/* { */
/*     double                cost; */
/*     ymir_mesh_t           *mesh = inverse_params->mesh; */
/*     ymir_pressure_elem_t  *press_elem = inverse_params->stokes_pc->stokes_op->press_elem; */
/*     ymir_vec_t            *uobs = inverse_params->uobs; */
/*     ymir_vec_t            *adjoint_vq = inverse_params->adjoint_vq; */
/*     ymir_vec_t            *old_up = ymir_vec_template (adjoint_vq); */
/*     ymir_dvec_t           *old_grad = ymir_vec_template (state->weak_vec); */
/*     ymir_dvec_t           *grad = ymir_dvec_new (mesh, 2, YMIR_GAUSS_NODE); */
/*     ymir_dvec_t           *grad_weak_factor = ymir_vec_template (state->weak_vec); */
/*     ymir_dvec_t           *grad_strain_exp  = ymir_vec_template (state->weak_vec); */
/*     ymir_dvec_t           *grad_yield_stress = ymir_vec_template (state->weak_vec); */
/*     ymir_dvec_t           *gradmass, *inv_grad_mass; */
/*     ymir_dvec_t           *weak_zone_stencil = inverse_params->weak_zone_stencil; */
/*     ymir_dvec_t           *upper_mantle_stencil = inverse_params->upper_mantle_stencil; */
/*     ymir_dvec_t           *weak_factor = ymir_vec_template (state->weak_vec); */
/*     ymir_dvec_t           *inc_update = ymir_dvec_new (mesh, 2, YMIR_GAUSS_NODE); */
/*     ymir_dvec_t           *ones = ymir_vec_template (state->weak_vec); */
/*     ymir_dvec_t           *upper_mantle = ymir_vec_template (state->weak_vec); */
/*     ymir_dvec_t           *grad_all = ymir_vec_template (grad); */
/*     ymir_dvec_t           *proj_mass = ymir_vec_template (state->weak_vec); */
/*     double                gradnorm, gradnormprev, orignorm = 0; */
/*     double                grad_proj[3], hessian[9], update[3]; */
/*     double                proj_grad_prefactor, proj_grad_strain_exp; */
/*     double                proj_grad_yield_stress; */
/*     double                state_rtol; */
/*     double                adjoint_rtol; */
/*     double                inc_rtol; */
/*     double                cg_rtol = 0.5; */
/*     double                cg_rtol_min; */
/*     double                cg_rtol_max; */
/*     double                lsalpha, lsmin, lsmax; */
/*     double                armijo_c1 = 1.0e-4; */
/*     double                expa = 1.6180339887; */
/*     int                   i = 0; */
/*     double                old_prefactor; */
/*     double                old_strain_exp; */
/*     double                old_yield_stress; */
/*     double                new_cost; */
/*     slabs_nl_stokes_problem_t *nl_stokes_resolve; */

/*     /\* For the optimization problem we will first compute the adjoint solution given */
/*        the surface observations and guessed solution.  */
/*        1) Guessed solution is stored in state->up */
/*        2) Observation is stored in inverse_params->uobs */
/*        3) Parameters(rheological) at current interation are in inverse_params->(n, prefactor, etc). */
/*     *\/ */


/*     /\* Initialize arrays for gradient, hessian, and update. Need to find a better way to do this *\/ */
    
/*     grad_proj[0] = 0; */
/*     grad_proj[1] = 0; */
/*     grad_proj[2] = 0; */

/*     hessian[0] = 0; */
/*     hessian[1] = 0; */
/*     hessian[2] = 0; */
/*     hessian[3] = 0; */
/*     hessian[4] = 0; */
/*     hessian[5] = 0; */
/*     hessian[6] = 0; */
/*     hessian[7] = 0; */
/*     hessian[8] = 0; */

    
/*     update[0] = 0; */
/*     update[1] = 0; */
/*     update[2] = 0; */


/*     /\* Compute adjoint solution where the RHS is the misfit in surface */
/*        velocities. */
/*     *\/ */
    
    

/*     YMIR_GLOBAL_PRODUCTION ("Entering Optimization Routine.\n"); */
/*     ymir_vec_set_zero (adjoint_vq); */
/*     YMIR_GLOBAL_PRODUCTION ("Set adjoint rhs to zero.\n"); */
    
    
/*     slabs_nonlinear_solver_solve_adjoint (adjoint_vq, state,  */
/* 					  uobs, lin_stokes, */
/* 					  &nl_stokes, press_elem, */
/* 					  physics_options, */
/* 					  discr_options, */
/* 					  solver_options, NULL, */
/* 					  NULL); */


/*     /\* Compute cost functional *\/ */
/*     cost = slabs_surface_vel_misfit (uobs, state, press_elem);     */
/*     YMIR_GLOBAL_PRODUCTIONF ("Optimization cost functional %E\n", cost); */
    
    
/*     /\* Compute the gradient: need to make option for different parameter inversion */
/*      *\/ */

/*     slabs_compute_gradient_weakfactor (grad_weak_factor, state, adjoint_vq, */
/*     				       physics_options, lin_stokes, &nl_stokes, */
/*     				       mesh, press_elem, NULL); */
    
/*     slabs_compute_gradient_strain_rate_exponent (grad_strain_exp, state, adjoint_vq, */
/*     						 physics_options, lin_stokes, &nl_stokes, */
/*     						 mesh, press_elem, NULL); */

/*     slabs_compute_gradient_yield_stress (grad_yield_stress, state, adjoint_vq, */
/*     					 physics_options, lin_stokes, &nl_stokes, */
/*     					 mesh, press_elem, NULL); */


    
/*     ymir_vec_set_value (ones, 1.0); */

/*     /\* Need to project gradient *\/ */
/*     proj_grad_prefactor = ymir_vec_innerprod (grad_weak_factor, weak_zone_stencil); */
/*     proj_grad_strain_exp = ymir_vec_innerprod (grad_strain_exp, upper_mantle_stencil); */
/*     proj_grad_yield_stress = ymir_vec_innerprod (grad_yield_stress, ones); */


/*     grad_proj[0] = -proj_grad_prefactor; */
/*     grad_proj[1] = -proj_grad_strain_exp; */
/*     grad_proj[2] = -proj_grad_yield_stress; */

/*     YMIR_GLOBAL_PRODUCTIONF ("gradient of prefactor: %g.\n", grad_proj[0]); */
/*     YMIR_GLOBAL_PRODUCTIONF ("gradient of strain exp: %g.\n", grad_proj[1]); */
/*     YMIR_GLOBAL_PRODUCTIONF ("gradient of yield stress: %g.\n", grad_proj[2]); */
	

    
/*     /\* Compute initial gradient norm *\/ */
/*     inv_grad_mass = ymir_vec_template (grad); */
/*     gradmass = ymir_vec_template (grad); */
/*     ymir_mass_lump (inv_grad_mass); */
/*     ymir_vec_reciprocal (inv_grad_mass); */
/*     ymir_vec_multiply (grad, inv_grad_mass, gradmass); */
/*     gradnorm = sqrt (ymir_vec_innerprod (gradmass, grad)); */
    
/*     /\* stats [11] = gradnorm; *\/ */

    
/*     double rel_tol = 1.0e-3; */
/*     double max_iter = 20; */
/*     double max_line_search = 40; */
/*     cg_rtol_max = 0.5; */
/*     cg_rtol_min = 2.0 * SC_EPS; */
/*     double state_rtol_max = 1.0e-6; */
/*     double state_rtol_min = 1.0e-6; */
/*     double adjoint_rtol_min = 1.0e-6; */
/*     double adjoint_rtol_max = 1.0e-6; */
/*     double inc_rtol_min = 1.0e-6; */
/*     double inc_rtol_max = 1.0e-6; */
/*     double factor; */
/*     double old_cg_rtol; */
    
/*     /\* Note this is the weak factor stencil: computed only once *\/ */
/*     slabs_physics_compute_weak_factor_distrib (weak_factor, physics_options); */
/*     slabs_physics_compute_UM_distrib (upper_mantle, physics_options); */
  
/*     YMIR_GLOBAL_VERBOSEF ("GRadient norm: %E\n", gradnorm); */
    
    
/*     for (i = 0; i < 25; i++) { */
      
/*       if (!i) { */
/* 	orignorm = gradnorm; */
/*       } */
/* #if 0 */
/*       /\* if the norm of the gradient is below the threshold *\/ */
/*       if (gradnorm < orignorm * rel_tol) { */
/* 	break; */
/*       } */
/* #endif */
/*       /\* if the maximum amount of iterations has be reached, exit *\/ */
/*       if (i >= max_iter) { */
/* 	break; */
/*       } */
      
/*       if (i) { */
/* 	factor = SC_MIN (1., gradnorm / orignorm); */
/* 	old_cg_rtol = cg_rtol; */
/* 	cg_rtol = */
/* 	  cg_rtol_max * pow (SC_MIN (1., gradnorm / gradnormprev), expa); */
/* 	old_cg_rtol = cg_rtol_max * pow (old_cg_rtol, expa); */
      
/* 	if (old_cg_rtol > 0.1 * cg_rtol_max * pow (cg_rtol_max, expa)) { */
/* 	  cg_rtol = SC_MAX (old_cg_rtol, cg_rtol); */
/* 	} */
      
/* 	cg_rtol = SC_MAX (cg_rtol, cg_rtol_min); */
/* 	YMIR_GLOBAL_VERBOSEF ("New cg rtol %E\n", cg_rtol); */

/* 	state_rtol = SC_MIN (state_rtol, state_rtol_max * factor); */
/* 	state_rtol = SC_MAX (state_rtol, state_rtol_min); */

/* 	adjoint_rtol = SC_MIN (adjoint_rtol, adjoint_rtol_max * factor); */
/* 	adjoint_rtol = SC_MAX (adjoint_rtol, adjoint_rtol_min); */

/* 	inc_rtol = SC_MIN (inc_rtol, inc_rtol_max * factor); */
/* 	inc_rtol = SC_MAX (inc_rtol, inc_rtol_min); */

/*       } */

/*       ymir_vec_set_zero (inc_update); */
/*       ymir_vec_set_zero (dprefactor); */
/*       ymir_vec_set_zero (dstrain_exp); */
/*       ymir_vec_set_zero (dyield_stress); */

/*       /\* if (i > 2){ *\/ */
/*       /\* 	cg_rtol = 1.0e-2; *\/ */
/*       /\* } *\/ */

/*       YMIR_GLOBAL_STATISTICSF ("Optimization iter %d, rtol %E\n", i, cg_rtol); */
/*       /\* Solve KKT system for descent direction *\/ */
/*       /\* slabs_cg_gauss_newton_hessian (inc_update, grad, cg_rtol, inverse_params); *\/ */
/*       /\* Explicitly form the Hessian *\/ */
/* #if 1 */
/*       slabs_hessian3d_construct_explicit (mesh, &hessian, inverse_params); */
/*       slabs_solve_hessian3d_direct (&update, &hessian, &grad_proj); */
/* #endif       */
/*       /\* If outer iteration > 1 then destroy previous adjoint/Newton operators and */
/* 	 preconditioners *\/ */
/*       if (i > 0){	       */
/*       slabs_nonlinear_stokes_op_pc_destroy (nl_stokes_resolve); */
/*       slabs_nonlinear_stokes_problem_destroy (nl_stokes_resolve); */
/*       } */
 

/*       /\* searchdotdescent is the inner product of the update from the Hessian and gradient *\/ */
/*       double search_dot_descent;  */
/*       /\* search_dot_descent = ymir_vec_innerprod (dprefactor, grad); *\/ */
/*       /\* YMIR_GLOBAL_VERBOSEF ("search/descent inner product %E\n", search_dot_descent); *\/ */
      
/*       #if 0 */
/*       if (searchdotdescent <= 0.) { */
/* 	/\* This shouldn't happen for G-N but, can be useful and should be implemented */
/* 	   for full Hessian *\/ */
/* 	YMIR_GLOBAL_LERROR ("Warning: search direction is not descent direction\n"); */
/*       } */
/*       #endif */
      
      
/*       /\* Need to save old information (velocity, pressure, etc) *\/ */
/*       old_strain_exp = physics_options->viscosity_stress_exponent; */
/*       old_prefactor = physics_options->toy_weakzone_plate1; */
/*       old_yield_stress = physics_options->viscosity_stress_yield; */
/*       lsalpha = 1.0; */
      
/*       /\* add the scaled update to the old parameters. First we must project the descent */
/* 	 direction to a single prefactor. We do this by taking the inner product of the stencil */
/* 	 of the weak zone and and descent direction. */
/*       *\/ */
/*       double proj_prefactor; */
/*       double proj_strain_exp; */
/*       double proj_yield_stress; */
/*       double proj_gradient_prefactor; */
/*       double proj_gradient_strain_exp; */
/*       double proj_gradient_yield_stress; */


/*       int line_search_count = 0; */
 
/*       /\* Projected values *\/ */
/*       proj_gradient_prefactor =  -grad_proj[0]; */
/*       proj_gradient_strain_exp = -grad_proj[1]; */
/*       proj_gradient_yield_stress = -grad_proj[2]; */


/*       YMIR_GLOBAL_PRODUCTIONF ("Update for the prefactor is: %g.\n", update[0]); */
/*       YMIR_GLOBAL_PRODUCTIONF ("Update for the strain rate exponent is: %g.\n", update[1]); */
/*       YMIR_GLOBAL_PRODUCTIONF ("Update for the yield stress is: %g.\n", update[2]); */


/*       do { */
/* 	double prefactor; */
/* 	double strain_exp; */
/* 	double yield_stress; */
/* 	double rescale = 1.0; */
/*         double    prefactor_scale = 1.0; */
/* 	double    strain_exp_scale = 1.0; */


/* 	prefactor = log (old_prefactor); */
/* 	prefactor += lsalpha * prefactor_scale   * update[0]; */
/* 	/\* need to transform the 'log prefactor' to exp(prefactor)' *\/ */
/* 	prefactor = exp (prefactor); */
/* 	YMIR_GLOBAL_PRODUCTIONF ("old prefactor is: %g, New prefactor is: %g.\n", old_prefactor, prefactor); */

/* 	strain_exp = log (old_strain_exp); */
/* 	strain_exp += lsalpha * strain_exp_scale * update[1]; */
/* 	/\* need to transform the 'log prefactor' to exp(prefactor)' *\/ */
/* 	strain_exp = exp (strain_exp); */
/* 	YMIR_GLOBAL_PRODUCTIONF ("old strain exp is: %g, New strain rate exponent is: %g.\n",  */
/* 				 old_strain_exp, strain_exp); */


/* 	yield_stress = log (old_yield_stress); */
/* 	yield_stress += lsalpha * update[2]; */
/* 	/\* need to transform the 'log prefactor' to exp(prefactor)' *\/ */
/* 	yield_stress = exp (yield_stress); */
/* 	YMIR_GLOBAL_PRODUCTIONF ("old yield stress is: %g, New yield stress is: %g.\n",  */
/* 				 old_yield_stress, yield_stress); */


/* 	/\* recompute weak zone. First we set the velocity-pressure vec to zero and change */
/* 	   the vale of the prefactor.  */
/* 	*\/ */
/*        	double prefactor_new = prefactor; */
/* 	double strain_exp_new = strain_exp; */
/* 	double yield_stress_new = yield_stress; */
/* 	ymir_vec_set_zero (state->vel_press_vec); */
/* 	slabs_update_prefactor_simple (prefactor_new, physics_options); */
/*  	slabs_update_strain_exp (strain_exp_new, physics_options); */
/* 	slabs_update_yield_stress (yield_stress_new, physics_options); */
/* 	slabs_physics_compute_weakzone (state->weak_vec, physics_options); */
  
/* 	/\* Create new stokes problem if we need to backtrack *\/ */
     
/* 	nl_stokes_resolve = slabs_nonlinear_stokes_problem_new (state, mesh, press_elem, */
/* 								physics_options); */
	
/* 	slabs_solve_stokes (lin_stokes, &nl_stokes_resolve, p8est, &mesh, &press_elem, */
/* 			    state, physics_options, discr_options, */
/* 			    solver_options, NULL, NULL, NULL); */
  
/* 	/\* Now compute new misfit in surface velocities *\/       */
/* 	new_cost = slabs_surface_vel_misfit (uobs, state, press_elem); */

/* 	YMIR_GLOBAL_VERBOSEF ("line search: alpha %f old cost %E new cost %E\n", */
/* 			      lsalpha, cost, new_cost); */
      

/* 	/\* Use globalization via Armijo Line search: If (J(m_k+1)<J(m_k)+c*alpha*d*g^T), we  */
/* 	   accept the new solution (as well as the operator used to compute next adjoint */
/* 	   solution). *\/ */
/* #if 0 */
/* 	if (new_cost < (cost - lsalpha  * armijo_c1 * ( prefactor_scale * proj_gradient_prefactor * proj_gradient_prefactor +  */
/* 							strain_exp_scale * proj_gradient_strain_exp * proj_gradient_strain_exp))){ */
/* #endif */

/* 	  if (new_cost < (cost - lsalpha * armijo_c1 * ( proj_gradient_prefactor * update[0] +  */
/* 							  proj_gradient_strain_exp * update[1] + */
/* 							 proj_gradient_yield_stress * update[2]))){ */


/* 	  YMIR_GLOBAL_PRODUCTIONF ("Cost function minimized in %i line searches for G-N iteration %i.\n",  */
/* 				   line_search_count, i); */

/* 	  /\* Since the misfit is minimized from this new parameter, we save the solution */
/* 	     (u,p) along with the adjoint operator, etc. Since the nl-stokes op is destroyed, we will */
/* 	     recompute it here. */
/* 	  *\/ */
	
/* 	  /\* Recompute the adjoint operator since it was destroyed. *\/ */
/* 	  slabs_nonlinear_stokes_op_new (nl_stokes_resolve, state, physics_options, 0, */
/* 					 solver_options->nl_solver_type, */
/* 					 solver_options->nl_solver_primaldual_type, */
/* 					 solver_options->nl_solver_primaldual_scal_type, NULL); */
  
/* 	  slabs_nonlinear_stokes_pc_new (nl_stokes_resolve, state, */
/* 					 solver_options->schur_diag_type, */
/* 					 solver_options->scaling_type, 0, NULL); */

	  
	  
/* 	  inverse_params->viscosity = nl_stokes_resolve->viscosity;  */
/*  	  inverse_params->prefactor = prefactor; */
/* 	  inverse_params->strain_exp = strain_exp; */
/* 	  inverse_params->yield_stress = yield_stress; */
/* 	  inverse_params->stokes_pc = nl_stokes_resolve->stokes_pc; */
/* 	  inverse_params->yielding_marker = nl_stokes_resolve->yielding_marker; */
/* 	  inverse_params->bounds_marker = nl_stokes_resolve->bounds_marker; */

/* 	  YMIR_GLOBAL_PRODUCTIONF (" The new prefactor is: %g.\n", inverse_params->prefactor); */
/* 	  YMIR_GLOBAL_PRODUCTIONF (" The new strain rate exponent is: %g.\n", inverse_params->strain_exp); */
/* 	  YMIR_GLOBAL_PRODUCTIONF (" The new yield stress is: %g.\n", inverse_params->yield_stress); */

/* 	  break; */

/* 	} */
/* 	else { */
/* 	  slabs_nonlinear_stokes_problem_destroy (nl_stokes_resolve); */
/* 	  lsalpha /= 2.0; */
/* 	  line_search_count++; */
/* 	} */
      
/*       } while ((max_line_search > line_search_count)); */
      
/*       /\* We assign the new cost that needs to be minimized *\/ */
/*       cost = new_cost; */
/*       gradnormprev = gradnorm; */
      
/*       /\* Compute the new gradient norm *\/ */
/*       ymir_vec_multiply (grad, inv_grad_mass, gradmass); */
/*       gradnorm = sqrt (ymir_vec_innerprod (gradmass, grad)); */


/*       YMIR_GLOBAL_PRODUCTIONF ("New compute optimization cost functional %E\n", cost); */
/*       YMIR_GLOBAL_PRODUCTIONF ("Entering Gauss-Newton iteration %d\n.", i+1); */

/*       /\* Compute the adjoint solution from the new surface velocities *\/ */
/*       ymir_vec_set_zero (adjoint_vq); */
/*       slabs_nonlinear_solver_solve_adjoint (adjoint_vq, state,  */
/* 					    uobs, lin_stokes, */
/* 					    &nl_stokes_resolve, press_elem, */
/* 					    physics_options, */
/* 					    discr_options, */
/* 					    solver_options, NULL, */
/* 					    NULL); */

/*       /\* Now compute the gradient to be used as the RHS for Hessian system *\/ */
/*       ymir_vec_set_zero (grad); */
/*       slabs_compute_gradient_weakfactor (grad_weak_factor, state, adjoint_vq, */
/*       					 physics_options, lin_stokes, */
/*       					 &nl_stokes_resolve, */
/*       					 mesh, press_elem, NULL); */
 

/*       slabs_compute_gradient_strain_rate_exponent (grad_strain_exp, state, adjoint_vq, */
/* 						   physics_options, lin_stokes,  */
/* 						   &nl_stokes_resolve, */
/*       						   mesh, press_elem, NULL); */
 

/*       slabs_compute_gradient_yield_stress (grad_yield_stress, state, adjoint_vq, */
/*       					   physics_options, lin_stokes, */
/*       					   &nl_stokes_resolve, */
/*       					   mesh, press_elem, NULL); */
 


/*       /\* Need to project gradient *\/ */
/*     proj_grad_prefactor = ymir_vec_innerprod (grad_weak_factor, weak_zone_stencil); */
/*     proj_grad_strain_exp = ymir_vec_innerprod (grad_strain_exp, upper_mantle_stencil); */
/*     proj_grad_yield_stress = ymir_vec_innerprod (grad_yield_stress, ones); */
    
/*     grad_proj[0] = -proj_grad_prefactor; */
/*     grad_proj[1] = -proj_grad_strain_exp; */
/*     grad_proj[2] = -proj_grad_yield_stress; */


/*     YMIR_GLOBAL_PRODUCTIONF ("gradient of prefactor: %g.\n", grad_proj[0]); */
/*     YMIR_GLOBAL_PRODUCTIONF ("gradient of strain exp: %g.\n", grad_proj[1]); */
/*     YMIR_GLOBAL_PRODUCTIONF ("gradient of strain yield stress: %g.\n", grad_proj[2]); */
	

    
/* } */

/*     /\* Final destruction of adjoint/Newton operators and nl-stokes problem *\/ */
/*     slabs_nonlinear_stokes_op_pc_destroy (nl_stokes_resolve); */
/*     slabs_nonlinear_stokes_problem_destroy (nl_stokes_resolve); */

  
/*     ymir_vec_destroy (old_up); */
/*     ymir_vec_destroy (old_grad); */
/*     ymir_vec_destroy (grad); */
/*     ymir_vec_destroy (grad_all); */
/*     ymir_vec_destroy (grad_weak_factor); */
/*     ymir_vec_destroy (grad_strain_exp); */
/*     ymir_vec_destroy (grad_yield_stress); */
/*     ymir_vec_destroy (gradmass); */
/*     ymir_vec_destroy (inv_grad_mass); */
/*     ymir_vec_destroy (weak_factor); */
/*     ymir_vec_destroy (upper_mantle); */
/*     ymir_vec_destroy (ones); */
/*     ymir_vec_destroy (inc_update); */
/*     ymir_vec_destroy (dprefactor); */
/*     ymir_vec_destroy (dstrain_exp); */
/*     ymir_vec_destroy (dyield_stress); */
/*     ymir_vec_destroy (proj_mass); */
    
/* } */



