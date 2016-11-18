/*
  This file is part of the ymir Library.
  ymir is a C library for modeling ice sheets

  Copyright (C) 2010, 2011 Carsten Burstedde, Toby Isaac, Georg Stadler,
                           Lucas Wilcox.

  The ymir Library is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
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

#ifndef SLABS_GRADIENT_H
#define SLABS_GRADIENT_H

#include <slabs_base.h>
#include <slabs_norm.h>
#include <slabs_physics_extended.h>
#include <slabs_discretization.h>
#include <slabs_discretization_extended.h>
#include <slabs_linear_stokes_problem.h>
#include <slabs_nonlinear_stokes_problem.h>
#include <slabs_io.h>
#include <slabs_vtk.h>
#include <slabs_setup.h>
#include <slabs_gradient.h>
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



/* enumerator for types of gradients */
typedef enum
{
  SL_GRADIENT_WEAKFACTOR,
  SL_GRADIENT_STRAIN_RATE_EXPONENT,
  SL_GRADIENT_UPPER_MANTLE_PREFACTOR,
  SL_GRADIENT_TRANSITION_ZONE_PREFACTOR,
  SL_GRADIENT_YIELD_STRESS,
  SL_GRADIENT_ACTIVATION_ENERGY
}
slabs_gradient_type_t;


/* Inverse Problems Data */
typedef enum
{
  SL_HESSIAN_SURFACE_VELOCITY,
  SL_HESSIAN_SURFACE_NORMAL_STRESS,
  SL_HESSIAN_SURFACE_STRAIN_RATE,
  SL_HESSIAN_SURFACE_VELOCITY_AND_NORMAL_STRESS,
  SL_HESSIAN_SURFACE_VELOCITY_AND_STRAIN_RATE,
  SL_HESSIAN_SURFACE_NORMAL_STRESS_AND_STRAIN_RATE,
  SL_HESSIAN_SURFACE_VELOCITY_NORMAL_STRESS_AND_STRAIN_RATE
}
slabs_hessian_data_t;

typedef enum
{
  SL_PRIOR_WEAKFACTOR,
  SL_PRIOR_STRAIN_RATE_EXPONENT,
  SL_PRIOR_UPPER_MANTLE_PREFACTOR,
  SL_PRIOR_TRANSITION_ZONE_PREFACTOR,
  SL_PRIOR_YIELD_STRESS,
  SL_PRIOR_ACTIVATION_ENERGY
}
slabs_priors_t;

typedef enum
{
  SL_PENALTY_VISCOSITY,
  SL_PENALTY_DEVIATORIC_STRESS
}
slabs_penalty_t;

typedef struct slabs_inverse_problem_obs
{
  ymir_vec_t *uobs;
  ymir_vec_t *IIe_obs;
  ymir_vec_t *topog_obs;
}
slabs_inverse_problem_obs_t;  



/* Inverse problem parameters */
typedef struct slabs_inverse_problem_params
{
  /* observations (surface velocities, residual topography,
     state of stress..) 
  */ 
  ymir_vec_t *up;
  ymir_vec_t *u;
  ymir_vec_t *IIe_misfit;
  ymir_vec_t *IIe_surf;
  ymir_vec_t *topog_misfit;
  ymir_vec_t *topog;
  sc_dmatrix_t *velocity_data;
  ymir_vec_t *velocity_data_vec;
  ymir_vec_t *IIe_obs;
  ymir_vec_t *obs_points_vel;
  ymir_vec_t *obs_points_IIe;
  double     *theta_data;
  double     *vel_data;
  double     average_viscosity_data;
  ymir_vec_t *uobs;
  ymir_vec_t *normal_stress_obs;
  ymir_vec_t *surface_normal_stress;
  ymir_vec_t *surface_IIe;

  /* physics options (rheological parameters, etc.) */
  double               strain_exp;
  double               prefactor;
  double               yield_stress;
  double               upper_mantle_prefactor;
  double               lower_mantle_prefactor;
  double               transition_zone_prefactor;
  double               activation_energy;
  double               domain_moment_of_inertia;
  ymir_dvec_t          *viscosity;
  ymir_vec_t           *viscosity_temperature;
  ymir_vec_t           *IIe;
  ymir_vec_t           *adjoint_vq;
  ymir_vel_dir_t       *vel_dir;
  ymir_dvec_t          *bounds_marker;
  ymir_dvec_t          *no_bounds_marker;
  ymir_dvec_t          *yielding_marker;
  ymir_dvec_t          *no_yielding_marker;
  ymir_dvec_t          *smooth_factor;
  ymir_vec_t           *surface_normals;
  ymir_vec_t           *surface_velocity_misfit_RHS;
  ymir_vec_t           *viscosity_mod;
  ymir_vec_t           *ones;
  ymir_vec_t           *log_IIe;
  ymir_vec_t           *log_viscosity;
  ymir_vec_t           *inverse_viscosity;
  ymir_vec_t           *deriv_viscosity_rhs;
  ymir_vec_t           *deriv_viscosity_param;
  ymir_vec_t           *viscosity_deriv_IIe;
  ymir_vec_t           *viscosity_second_deriv_IIe;
  ymir_vec_t           *viscosity_average_rhs;
  ymir_vec_t           *sam_weakzone_stencil;
  ymir_vec_t           *izu_weakzone_stencil;
  ymir_vec_t           *ryu_weakzone_stencil;
  ymir_mesh_t          *mesh;
  ymir_pressure_elem_t *press_elem;
  ymir_stokes_op_t     *stokes_op;
  ymir_stokes_pc_t     *stokes_pc;
  /* Cost funtional information */
  double            misfit_plate_vel;
  double            misift_residual_topog;
  double            misfit_strain_rate_IIe;
  double            average_visc_area;
  double            average_visc_area_misfit_cost;
  double            average_visc_area_misfit;
  double            gradient_visc_area;
  /* Gradient Check options */

  /* optimization vectors (gradient, incrementals, etc. */
  int          gradient_type;
  ymir_dvec_t *grad_weak_factor;
  ymir_dvec_t *grad_strain_exp;
  ymir_dvec_t *grad_yield_stress;
  ymir_dvec_t *grad_activation_energy;
  ymir_dvec_t *grad_UM_prefactor;
  ymir_dvec_t *grad_LM_prefactor;
  ymir_dvec_t *grad_TZ_prefactor;
  ymir_dvec_t *euev;
  ymir_dvec_t *eu;
  ymir_dvec_t *ev;
  ymir_vec_t  *viscosity_prior;
  
  double      grad_weak_factor_proj;
  double      grad_strain_exp_proj;
  double      grad_yield_stress_proj;
  double      grad_activation_energy_proj;
  double      grad_upper_mantle_prefactor_proj;
  double      grad_transition_zone_prefactor_proj;
  double      grad_lower_mantle_prefactor_proj;
  double      grad_weak_factor_visc_avg;
  double      grad_strain_exp_visc_avg;
  double      grad_yield_stress_visc_avg;
  double      grad_activation_energy_visc_avg;
  double      grad_upper_mantle_prefactor_visc_avg;
  double      grad_transition_zone_prefactor_visc_avg;
  double      grad_lower_mantle_prefactor_visc_avg;

  int    compute_gradient_vel_weakfactor;
  int    compute_gradient_vel_strain_rate_exponent;
  int    compute_gradient_vel_yield_stress;
  int    compute_gradient_vel_activation_energy;
  int    compute_gradient_vel_UM_prefactor;
  int    compute_gradient_vel_TZ_prefactor;
  int    compute_gradient_viscosity_weakfactor;
  int    compute_gradient_viscosity_strain_rate_exponent;
  int    compute_gradient_viscosity_yield_stress;
  int    compute_gradient_viscosity_activation_energy;
  int    compute_gradient_viscosity_UM_prefactor;
  int    compute_gradient_viscosity_TZ_prefactor;




  /* gradient descent parameters */
  double      alpha;
  double      scaled_misfit;

  /* Viscosity gradient w.r.t rheological parameters */
  ymir_vec_t *grad_viscosity_weakfactor;
  ymir_vec_t *grad_viscosity_strain_rate_exponent;
  ymir_vec_t *grad_viscosity_yield_stress;
  ymir_vec_t *grad_viscosity_activation_energy;
  ymir_vec_t *grad_viscosity_UM_prefactor;
  ymir_vec_t *grad_viscosity_TZ_prefactor;

  /* Stencils */
  ymir_dvec_t *upper_mantle_marker;
  ymir_dvec_t *lower_mantle_marker;
  ymir_dvec_t *weakzone_marker;
  ymir_dvec_t *transition_zone_marker;
  ymir_dvec_t *viscosity_stencil;

  /* Parameter distributions */
  ymir_dvec_t *upper_mantle_distrib;
  ymir_dvec_t *lower_mantle_distrib;
  ymir_dvec_t *strain_rate_exp_distrib;
  ymir_dvec_t *weakzone_distrib;
  ymir_dvec_t *transition_zone_distrib;
  ymir_dvec_t *yield_stress_distrib;
  ymir_dvec_t *viscosity_distrib;


  /* incremental solve tolerances */
  double inc_rtol_max;
  double inc_rtol_min;
  double inc_rtol;
  int inc_maxiter;

  /* regularization for tikhonov */
  double reg_tik_prefactor;
  double reg_tik_strain_exp;
  double reg_tik_yield_stress;

  /* weighting for priors */
  double prior_weakfactor_scale;
  double prior_strain_rate_exponent_scale;
  double prior_yield_stress_scale;
  double prior_viscosity;
  double prior_upper_mantle_prefactor_scale;
  double prior_transition_zone_prefactor_scale;
  double prior_activation_energy_scale;

  int prior_weakfactor;
  int prior_strain_rate_exponent;
  int prior_yield_stress;
  int prior_upper_mantle_prefactor;
  int prior_transition_zone_prefactor;
  int prior_activation_energy;

  ymir_vec_t *weakzone_factor_prior_misfit;
  ymir_vec_t *strain_rate_exponent_prior_misfit;
  ymir_vec_t *activation_energy_prior_misfit;
  ymir_vec_t *yield_stress_prior_misfit;
  ymir_vec_t *upper_mantle_prefactor_prior_misfit;
  ymir_vec_t *transition_zone_prefactor_prior_misfit;

  /* weighting for Tikhonov */
  double tik_prefactor;
  double tik_strain_exp;
  double tik_yield_stress;

  /* information for krylov tolerances */
  double krylov_rtol;
  double krylov_atol;
  int krylov_gmres_num_vecs;
  int krylov_maxiter;
  int krylov_num_iter;

  /* information for PCG solution of Hessian system */
  int    cg_solve_kyrlov_count;
  int    cg_step_krylove_count;
  int    cg_solve_steps;
  

  /* misfit for priors */
  ymir_vec_t *prior_weakzone_misfit;
  ymir_vec_t *prior_upper_mantle_misfit;
  ymir_vec_t *prior_lower_mantle_misfit;
  ymir_vec_t *prior_transition_zone_misfit;
  ymir_vec_t *prior_strain_rate_exp_misfit;
  ymir_vec_t *prior_yield_stress_misfit;
  ymir_vec_t *prior_viscosity_misfit;

 
  /* line search parameters and variables */
  double cost;
  double c;

  /* incremental update (global values) */
  double inc_update_strain_rate_exp;
  double inc_update_yield_stress;
  double inc_update_UM_prefactor;
  double inc_update_TZ_prefactor;
  double inc_update_activation_energy;
  double inc_update_weakzone_prefactor;

  /* incremental update (vector) */
  ymir_vec_t *inc_update_strain_rate_exp_vec;
  ymir_vec_t *inc_update_yield_stress_vec;
  ymir_vec_t *inc_update_UM_prefactor_vec;
  ymir_vec_t *inc_update_TZ_prefactor_vec;
  ymir_vec_t *inc_update_activation_energy_vec;
  ymir_vec_t *inc_update_weakzone_prefactor_vec;

}
slabs_inverse_problem_params_t;


slabs_inverse_problem_params_t *
slabs_inverse_problem_new (ymir_mesh_t *mesh, slabs_stokes_state_t *state,
			   ymir_pressure_elem_t *press_elem);

void
slabs_inverse_problem_clear (slabs_inverse_problem_params_t *inverse_problem_grad);

sc_dmatrix_t *
slabs_inverse_params_init_load_velocity_data (slabs_inverse_problem_params_t *inverse_params,
					      mangll_cnodes_t *cnodes);

ymir_vec_t *
slabs_inverse_params_init_velocity_data_vec (slabs_inverse_problem_params_t *inverse_params,
					     ymir_mesh_t *mesh);

void
slabs_inverse_destroy_data (slabs_inverse_problem_params_t *inverse_params);

void
slabs_initial_guess_weakfactors (slabs_physics_options_t *physics_options);
  

static void 
slabs_update_weakfactors_x_section (double sam_weakfactor, double izu_weakfactor,
				    double ryu_weakfactor,
				    slabs_physics_options_t *physics_options);
static void
slabs_update_strain_exp (double strain_rate_exp_update,
			 slabs_physics_options_t *physics_options);

static void 
slabs_update_yield_stress (double yield_stress_update,
			   slabs_physics_options_t *physics_options);

void
slabs_data_load_velocity (ymir_mesh_t *mesh,
			  ymir_vec_t *vel_data,
			  slabs_physics_options_t *physics_options);
  
void
slabs_inverse_params_plot_velocity (slabs_inverse_problem_params_t *inverse_params,
				    const char *vtk_filepath);


void
slabs_inverse_problem_destroy (slabs_inverse_problem_params_t *inverse_params);

void
slabs_initialize_inverse_params (slabs_inverse_problem_params_t *inverse_params,
				 slabs_nl_stokes_problem_t *nl_stokes,
				 slabs_nl_solver_options_t *solver_options,
				 slabs_physics_options_t *physics_options);

void 
slabs_gradient_parameters (const slabs_gradient_type_t *grad_param, slabs_inverse_problem_params_t *inverse_params);

void 
slabs_compute_gradient_weakfactor (slabs_inverse_problem_params_t *inverse_params,
				   const char *vtk_filepath);

void 
slabs_compute_gradient_strain_rate_exponent (slabs_inverse_problem_params_t *inverse_params,
					     const char *vtk_filepath);
void 
slabs_compute_gradient_yield_stress (slabs_inverse_problem_params_t *inverse_params,
				     const char *vtk_filepath);

void 
slabs_compute_gradient_activation_energy (slabs_inverse_problem_params_t *inverse_params,
					  const char *vtk_filepath);

void 
slabs_compute_gradient_UM_prefactor (slabs_inverse_problem_params_t *inverse_params,
				     const char *vtk_filepath);

void 
slabs_compute_gradient_transition_zone_prefactor (slabs_inverse_problem_params_t *inverse_params,
						  const char *vtk_filepath);

void
slabs_read_data (double *vel_data, double *theta,
		 slabs_physics_options_t *physics_options, 
		 slabs_inverse_problem_params_t *inverse_params);

		 
void
slabs_read_1Dvelocity_data (const char *textfile, double *theta,
			    double *vel);


void
slabs_velocity_vec_data2D (ymir_vec_t *vel, ymir_vec_t *obs_points,
			   slabs_physics_options_t *physics_options,
			   slabs_inverse_problem_params_t *inverse_params,
			   const char *vtk_filepath);

static inline 
void ComputeStressEigenVector(double A[3][3], double Q[3][3], double w[3]);

void
slabs_compute_stress_axis (double B[3][3]);

void
slabs_stress_tensor_output (ymir_vec_t *stress_tensor,
			    const char *vtk_filepath);
			    //			    slabs_inverse_problem_params_t *inverse_params);

void
slabs_solve_optimization (slabs_inverse_problem_params_t *inverse_params);



void 
slabs_compute_weakzone_prior (slabs_inverse_problem_params_t *inverse_params,
				       const char *vtk_filepath);

void 
slabs_compute_strain_rate_exponent_prior (slabs_inverse_problem_params_t *inverse_params,
					  const char *vtk_filepath);

void 
slabs_compute_yield_stress_prior (slabs_inverse_problem_params_t *inverse_params,
				  const char *vtk_filepath);

void 
slabs_compute_activation_energy_prior (slabs_inverse_problem_params_t *inverse_params,
				       const char *vtk_filepath);

void
slabs_log_viscosity (slabs_nl_stokes_problem_t *nl_stokes,
		     const char *vtk_filepath);


void
slabs_setup_mesh_data_weakzone (slabs_inverse_problem_params_t *inverse_params,
				slabs_stokes_state_t *state,
				slabs_physics_options_t *physics_options,
				const char *vtk_filepath);


void
slabs_weakzone_stencils (slabs_inverse_problem_params_t *inverse_params);

void 
slabs_clear_inverse_problem_empty (slabs_inverse_problem_params_t *inverse_params);

void
slabs_initialize_weakzone_stencils (slabs_inverse_problem_params_t *inverse_params);

void
slabs_destroy_weakzone_stencils (slabs_inverse_problem_params_t *inverse_params);

#endif /* SLABS_GRADIENT_H */
