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
static void
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
  /* ymir_vec_copy (rhs_inc, vq_out); */
  ymir_stress_op_destroy (stress_op_mod1);
  ymir_stress_op_destroy (stress_op_mod2);
  ymir_stress_op_destroy (stress_op_mod3);

  

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);
  sc_dmatrix_destroy (viscosity_second_deriv_IIe_el_mat);
  sc_dmatrix_destroy (eueu_inc_el_mat);
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
  
  ymir_vec_multiply_in1 (grad_viscosity_strain_rate_exponent, out);


}


void
slabs_hessian_block_mm_upper_mantle (slabs_inverse_problem_params_t *inverse_params,
				     slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *euev = inverse_params->euev;
  ymir_vec_t  *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_vec_t  *grad_viscosity_upper_mantle = inverse_params->grad_viscosity_UM_prefactor;
  ymir_vec_t  *out = ymir_vec_clone (euev);
  
  ymir_vec_multiply_in1 (grad_viscosity_upper_mantle, out);


}

void
slabs_hessian_block_mm_transition_zone (slabs_inverse_problem_params_t *inverse_params,
					slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *euev = inverse_params->euev;
  ymir_vec_t  *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_vec_t  *grad_viscosity_transition_zone = inverse_params->grad_viscosity_TZ_prefactor;
  ymir_vec_t  *out = ymir_vec_clone (euev);
  
  ymir_vec_multiply_in1 (grad_viscosity_transition_zone, out);


}
				   

void
slabs_hessian_block_mm_weakfactor (slabs_inverse_problem_params_t *inverse_params,
				   slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *euev = inverse_params->euev;
  ymir_vec_t  *weakzone_marker = inverse_params->weakzone_marker;
  ymir_vec_t  *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t  *out = ymir_vec_clone (euev);
  
  ymir_vec_multiply_in1 (grad_viscosity_weakfactor, out);


}

void
slabs_hessian_block_mm_yield_stress (slabs_inverse_problem_params_t *inverse_params,
				     slabs_hessian_params_t *hessian_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t  *euev = inverse_params->euev;
  ymir_vec_t  *yielding_marker = inverse_params->yielding_marker;
  ymir_vec_t  *grad_viscosity_yield_stress = inverse_params->grad_viscosity_yield_stress;
  ymir_vec_t  *out = ymir_vec_clone (euev);
  
  ymir_vec_multiply_in1 (grad_viscosity_yield_stress, out);


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


static void 
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



static void 
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


static void 
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


static void 
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

static void 
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


static void 
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

