 
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



 
typedef struct gradient_fn_data
{
  ymir_dvec_t *grad_weak_factor;
  ymir_dvec_t *grad_strain_exp;
  ymir_dvec_t *grad_TZ_prefactor;
  ymir_dvec_t *grad_yield_stress;
  ymir_dvec_t *grad_UM_prefactor;
  ymir_dvec_t *grad_activation_energy;
}
slabs_grad_data_t;



slabs_inverse_problem_params_t *
slabs_inverse_problem_new (ymir_mesh_t *mesh,
			   slabs_stokes_state_t *state,
			   ymir_pressure_elem_t *press_elem)
{
  slabs_inverse_problem_params_t *inverse_problem_grad;
  const char    *this_fn_name = "creating new inverse problem";
    
  YMIR_GLOBAL_INFOF ("Done %s\n", this_fn_name);


  inverse_problem_grad = YMIR_ALLOC (slabs_inverse_problem_params_t, 1);
   
  inverse_problem_grad->mesh = mesh;
  inverse_problem_grad->press_elem = press_elem;
 
  /* assign pointers to misfit and adjoint velocitiy and pressure */
  inverse_problem_grad->adjoint_vq = ymir_vec_template (state->vel_press_vec);
  /* inverse_problem_grad->surface_velocity_misfit = NULL; */
  /* inverse_problem_grad->surface_normal_stress_misfit = NULL; */
  /* inverse_problem_grad->surface_IIe_misfit = NULL; */
  
  /* assign pointers to gradient vectors */
  inverse_problem_grad->grad_weak_factor = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_problem_grad->grad_strain_exp = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_problem_grad->grad_yield_stress = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_problem_grad->grad_TZ_prefactor = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_problem_grad->grad_UM_prefactor = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_problem_grad->grad_activation_energy = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
 
  /* assign pointers to incremental update directions */
  inverse_problem_grad->inc_update_strain_rate_exp_vec = NULL;
  inverse_problem_grad->inc_update_yield_stress_vec = NULL;
  inverse_problem_grad->inc_update_UM_prefactor_vec = NULL;
  inverse_problem_grad->inc_update_TZ_prefactor_vec = NULL;
  inverse_problem_grad->inc_update_activation_energy_vec = NULL;
  inverse_problem_grad->inc_update_weakzone_prefactor_vec = NULL;


  /* assign pointers to rheological structures */
  inverse_problem_grad->viscosity = NULL;
  inverse_problem_grad->bounds_marker = NULL;
  inverse_problem_grad->yielding_marker = NULL;
  

  /* assign pointers to data */
  inverse_problem_grad->normal_stress_obs = NULL;
  inverse_problem_grad->IIe_obs = NULL;


  /* assign pointers to data */
  inverse_problem_grad->velocity_data = NULL;
  inverse_problem_grad->velocity_data_vec = NULL;
  inverse_problem_grad->stokes_pc = NULL;
  inverse_problem_grad->stokes_op = NULL;

  inverse_problem_grad->surface_normal_stress = NULL;
  inverse_problem_grad->surface_IIe = NULL;
  

  return inverse_problem_grad;

}

void
slabs_initialize_gradient_params (slabs_inverse_problem_params_t *inverse_params,
				  slabs_nl_stokes_problem_t *nl_stokes,
				  slabs_nl_solver_options_t *solver_options,
				  slabs_physics_options_t *physics_options)
{
  const char *this_fn_name = "Initializing adjoing solve information.\n";

  /* Initialize gradient structures */
  
  YMIR_GLOBAL_PRODUCTION ("Setting the inverse parameters.\n");
  inverse_params->stokes_pc = nl_stokes->stokes_pc;
  inverse_params->viscosity = nl_stokes->viscosity;
  inverse_params->bounds_marker = nl_stokes->bounds_marker;
  inverse_params->yielding_marker = nl_stokes->yielding_marker;
  inverse_params->strain_exp = physics_options->viscosity_stress_exponent;
  inverse_params->yield_stress = physics_options->viscosity_stress_yield; 
  inverse_params->vel_dir = nl_stokes->vel_dir;
  inverse_params->krylov_rtol = solver_options->krylov_rtol;
  inverse_params->krylov_atol = solver_options->krylov_atol;
  inverse_params->krylov_gmres_num_vecs = solver_options->krylov_gmres_num_vecs;
  inverse_params->krylov_maxiter = solver_options->krylov_maxiter;
  
  YMIR_GLOBAL_INFOF ("Done with %s", this_fn_name);

}



void
slabs_inverse_problem_clear (slabs_inverse_problem_params_t *inverse_problem_grad)
{
  const char    *this_fn_name = "clearing variables";
  slabs_inverse_destroy_data (inverse_problem_grad);

  ymir_vec_destroy (inverse_problem_grad->grad_weak_factor);
  ymir_vec_destroy (inverse_problem_grad->grad_strain_exp);
  ymir_vec_destroy (inverse_problem_grad->grad_yield_stress);
  ymir_vec_destroy (inverse_problem_grad->grad_TZ_prefactor);
  ymir_vec_destroy (inverse_problem_grad->grad_UM_prefactor);
  ymir_vec_destroy (inverse_problem_grad->grad_activation_energy);
  ymir_vec_destroy (inverse_problem_grad->adjoint_vq);
  YMIR_GLOBAL_INFOF ("Done %s\n", this_fn_name);


}

sc_dmatrix_t *
slabs_inverse_params_init_velocity_data (slabs_inverse_problem_params_t *inverse_params,
					 mangll_cnodes_t *cnodes)
{
  const mangll_locidx_t  n_nodes = cnodes->Ncn;
  const char    *this_fn_name = "creating velocity data structure";

  /* check state */
  YMIR_ASSERT (inverse_params->velocity_data == NULL);

  /* create temperature field */
  inverse_params->velocity_data = sc_dmatrix_new (n_nodes, 1);

  YMIR_GLOBAL_INFOF ("Done %s.\n", this_fn_name);
  /* return temperature field */
  return inverse_params->velocity_data;
}

ymir_vec_t *
slabs_inverse_params_init_velocity_data_vec (slabs_inverse_problem_params_t *inverse_params,
					     ymir_mesh_t *mesh)
{
  const char    *this_fn_name = "creating velocity data vector";

  /* check state */
  YMIR_ASSERT (inverse_params->velocity_data != NULL);

  /* create vector view for temperature */
  inverse_params->velocity_data_vec = ymir_cvec_new_data (mesh, 1, inverse_params->velocity_data);

  YMIR_GLOBAL_INFOF ("Done %s.\n", this_fn_name);

  /* return temperature vector */
  return inverse_params->velocity_data_vec;

}


void
slabs_inverse_destroy_data (slabs_inverse_problem_params_t *inverse_params)
{
  const char    *this_fn_name = "destroying velocity data";
  
  if (inverse_params->velocity_data != NULL) {
    sc_dmatrix_destroy (inverse_params->velocity_data);
    inverse_params->velocity_data = NULL;
  }
  if (inverse_params->velocity_data_vec != NULL) {
    ymir_vec_destroy (inverse_params->velocity_data_vec);
    inverse_params->velocity_data_vec = NULL;
  }
  YMIR_GLOBAL_INFOF ("Done %s\n", this_fn_name);

}

void
slabs_inverse_problem_destroy (slabs_inverse_problem_params_t *inverse_params)
{
  const char    *this_fn_name = "destroy data";

  slabs_inverse_problem_clear (inverse_params);
  YMIR_FREE (inverse_params);
  YMIR_GLOBAL_INFOF ("Done %s\n", this_fn_name);

}




static void 
slabs_initial_guess_weakfactors (slabs_physics_options_t *physics_options)

{
  const char    *this_fn_name = "changing parameters to initial guess";
  double        weakzone_plate_sam;
  double        weakzone_plate_izu;
  double        weakzone_plate_ryu;
  double        weakzone_guess = 1.0e-5;


  YMIR_GLOBAL_PRODUCTIONF ("Into %s\n", this_fn_name);

  physics_options->weakzone_import_weak_factor_sam = weakzone_guess;
  physics_options->weakzone_import_weak_factor_izu = weakzone_guess;
  physics_options->weakzone_import_weak_factor_ryu = weakzone_guess;

  YMIR_GLOBAL_PRODUCTIONF ("Guessed weak zone is: %g.\n", physics_options->weakzone_import_weak_factor_sam);

}

static void 
slabs_update_weakfactors_x_section (double sam_weakfactor, double izu_weakfactor,
				    double ryu_weakfactor,
				    slabs_physics_options_t *physics_options)

{
  const char    *this_fn_name = "Updating weakfactor";

  YMIR_GLOBAL_PRODUCTIONF ("Into %s\n", this_fn_name);

  /* redefine weakzone values from initial guesses */
  physics_options->weakzone_import_weak_factor_sam = sam_weakfactor;
  physics_options->weakzone_import_weak_factor_izu = izu_weakfactor;
  physics_options->weakzone_import_weak_factor_ryu = ryu_weakfactor;

  YMIR_GLOBAL_PRODUCTIONF ("Guessed weak zone is: %g.\n", physics_options->weakzone_import_weak_factor_sam);

}


static void
slabs_update_strain_exp (double strain_rate_exp_update,
			 slabs_physics_options_t *physics_options)
				  
{
  const char *this_fn_name = "Updating strain rate exponent";

  YMIR_GLOBAL_PRODUCTIONF (" Into %s\n", this_fn_name);
  /* reset value */
  physics_options->viscosity_stress_exponent = strain_rate_exp_update;
  YMIR_GLOBAL_PRODUCTIONF (" New strain rate exponent is: %g.\n", physics_options->viscosity_stress_exponent);


}


static void 
slabs_update_yield_stress (double yield_stress_update,
			   slabs_physics_options_t *physics_options)

{
  double yield_stress_old = physics_options->viscosity_stress_yield;
  double yield_stress_new = yield_stress_old;
  const char *this_fn_name = "Updating yield stress";
  
  YMIR_GLOBAL_PRODUCTIONF ("Into %s\n.", this_fn_name);
  /* reset value for yield stress */
  physics_options->viscosity_stress_yield = yield_stress_update;

}

void
slabs_inverse_params_load_velocity_data (slabs_inverse_problem_params_t *inverse_params, ymir_mesh_t *mesh,
					 const char *filepath_filename_import_txt,
					 const char *filepath_filename_import_bin)
{
  mangll_t           *mangll = mesh->ma;
  mangll_cnodes_t    *cnodes = mesh->cnodes;
  const char *this_fn_name = "loading and creating velocity data.\n";

  /* check input */
  YMIR_ASSERT (inverse_params->velocity_data == NULL);

  /* initialize temperature in Stokes state */
  slabs_inverse_params_init_velocity_data (inverse_params, cnodes);

  /* read temperature from file into Stokes state */
  slabs_io_read_temperature (inverse_params->velocity_data, mangll, mesh->cnodes,
                             filepath_filename_import_txt,
                             filepath_filename_import_bin);


  /* initialize temperature vector in Stokes state */
  slabs_inverse_params_init_velocity_data_vec (inverse_params, mesh);
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);
}


static void
slabs_surface_velocity_viz (slabs_stokes_state_t *state,
			    ymir_pressure_elem_t *press_elem,
			    const char *vtk_filepath,
			    slabs_physics_options_t *physics_options)
  
{
  ymir_mesh_t         *mesh = state->vel_press_vec->mesh;
  ymir_vec_t          *up = state->vel_press_vec;
  ymir_vec_t          *u = ymir_cvec_new (mesh,3);
  ymir_vec_t          *surface_vel = ymir_face_cvec_new (mesh, SL_TOP, 3);
  char                path[BUFSIZ];
    
  ymir_stokes_vec_get_velocity (up, u, press_elem);
  ymir_interp_vec (u, surface_vel);
  /* Write out stencil to vtk file */
  snprintf (path, BUFSIZ, "%s_surface_velocity", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  surface_vel, "surface_vel",
		  NULL);

  ymir_vec_destroy (surface_vel);
  ymir_vec_destroy (u);

}


void
slabs_inverse_params_plot_velocity (slabs_inverse_problem_params_t *inverse_params,
				    const char *vtk_filepath)
{
  ymir_mesh_t         *mesh = inverse_params->mesh;
  ymir_vec_t          *surface_vel = ymir_face_cvec_new (mesh, SL_TOP, 1);
  char                path[BUFSIZ];
    
  ymir_interp_vec (inverse_params->velocity_data_vec, surface_vel);
  /* Write out stencil to vtk file */
  snprintf (path, BUFSIZ, "%s_surface_velocity_data", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  surface_vel, "surface_vel",
		  NULL);

  ymir_vec_destroy (surface_vel);

				    
}

 /* Compute surface velocities misfit */
 static double
 slabs_surface_vel_misfit (slabs_inverse_problem_params_t *inverse_params)
	
 {
   ymir_mesh_t     *mesh = inverse_params->mesh;
   ymir_vec_t      *u = inverse_params->u;
   ymir_cvec_t     *tmp = ymir_vec_clone (u);
   ymir_vec_t      *uobs = inverse_params->velocity_data_vec;
   ymir_vec_t      *surface_velocity_misfit_RHS = inverse_params->surface_velocity_misfit_RHS;
   
 
   /* need to compute misfit uobs-u */
   ymir_vec_add (-1.0, uobs, tmp);

   /* now apply mass matrix  M(uobs-u) */
   ymir_mass_apply (tmp, surface_velocity_misfit_RHS);

   /* compute full misfit term 0.5 ||uobs-u||^2 weighted with Mass matrix */
   inverse_params-> misfit_plate_vel = 0.5 * ymir_vec_innerprod (surface_velocity_misfit_RHS, 
									tmp);

   /* destroy created parameters */
   ymir_vec_destroy (tmp);

   return inverse_params->misfit_plate_vel;
 }

#if 0
void 
slabs_gradient_parameters (const slabs_gradient_type_t *grad_param, slabs_inverse_problem_params_t *inverse_params)
{
  switch (inverse_params->gradient_type) {
  case SL_GRADIENT_WEAKFACTOR:
    slabs_compute_gradient_weakfactor (inverse_params, NULL);
    break;
    
  case SL_GRADIENT_STRAIN_RATE_EXPONENT:
    slabs_compute_gradient_strain_rate_exponent (inverse_params, NULL);
    break;

  case SL_GRADIENT_UPPER_MANTLE_PREFACTOR:
    slabs_compute_gradient_UM_prefactor (inverse_params, NULL);
    break;

  case SL_GRADIENT_TRANSITION_ZONE_PREFACTOR:
    slabs_compute_gradient_transition_zone_prefactor (inverse_params, NULL);
    break;

  case SL_GRADIENT_YIELD_STRESS:
    slabs_compute_gradient_yield_stress (inverse_params, NULL);
    break;

  case SL_GRADIENT_ACTIVATION_ENERGY:
    slabs_compute_gradient_activation_energy (inverse_params, NULL);
    break;
  }
}
 
#endif

void 
slabs_compute_gradient_weakfactor (slabs_inverse_problem_params_t *inverse_params,
				   const char *vtk_filepath)

{
  const char *this_fn_name =  "Computing the gradient of a weakfactor.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *no_yielding = inverse_params->no_yielding_marker;
  ymir_vec_t  *bounds_marker = inverse_params->bounds_marker;
  ymir_dvec_t *grad_weak_factor = inverse_params->grad_weak_factor;
  ymir_dvec_t *grad_mass = ymir_vec_template (grad_weak_factor);
  ymir_dvec_t *euev = ymir_vec_clone (inverse_params->euev); 
  ymir_dvec_t *viscosity_mod = inverse_params->viscosity_mod; 
  ymir_dvec_t *smooth_factor = inverse_params->smooth_factor; 
  ymir_dvec_t *ones = inverse_params->ones;
  double grad_proj = inverse_params->grad_weak_factor_proj;
  char                path[BUFSIZ];



  /* adjust for new weakfactor vector */
  
  ymir_dvec_multiply_in (viscosity_mod, euev);
  ymir_dvec_multiply_in1 (no_yielding, euev);
  ymir_dvec_multiply_in (smooth_factor, euev);

  /* Compute integral */
  
  
  ymir_mass_apply (euev, grad_weak_factor);
  grad_proj =  ymir_vec_innerprod (grad_mass, ones);
  YMIR_GLOBAL_PRODUCTIONF ("gradient of prefactor is: %g\n.", grad_proj);  


  ymir_vtk_write (mesh, path,
  		  grad_weak_factor, "gradient_weak_factor",
  		  smooth_factor, "smooth_factor",
  		  NULL);

  ymir_vec_copy (grad_mass, grad_weak_factor);

  /* Destroy vectors */
  ymir_vec_destroy (euev);
  ymir_vec_destroy (grad_mass);
#endif
  YMIR_GLOBAL_INFOF ("Finished %s", this_fn_name);
  
}

void 
slabs_compute_gradient_strain_rate_exponent (slabs_inverse_problem_params_t *inverse_params,
					     const char *vtk_filepath)


{
  const char *this_fn_name = "Computing the gradient of the strain rate exponent.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_dvec_t *bounds_marker = inverse_params->bounds_marker;
  ymir_dvec_t *no_yielding_marker = inverse_params->no_yielding_marker;
  ymir_dvec_t *log_IIe = inverse_params->log_IIe;
  ymir_dvec_t *euev = ymir_vec_clone (inverse_params->euev);
  ymir_dvec_t *viscosity_mod = ymir_vec_clone (inverse_params->viscosity_mod);
  ymir_dvec_t *grad_strain_exp = inverse_params->grad_strain_exp;
  double n = inverse_params->strain_exp;
  double log_factor = -0.5 / (n);
  double power = (1 - n)/ (n + n);
  char                path[BUFSIZ];

  
  YMIR_GLOBAL_PRODUCTIONF ("The strain rate exponent is: %E, and the log factor is: %E\n.", 
			   n, log_factor);
  
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
    
  ymir_dvec_multiply_in (log_IIe, euev);
  ymir_dvec_scale (log_factor, euev);
  ymir_vec_multiply_in1 (viscosity_mod, euev);
  ymir_vec_multiply_in1 (no_yielding_marker, euev);
  ymir_vec_multiply_in1 (bounds_marker, euev);
  ymir_vec_multiply_in1 (upper_mantle_marker, euev);

  
  ymir_mass_apply (euev, grad_strain_exp);
  inverse_params->grad_strain_exp_proj = ymir_vec_innerprod (grad_strain_exp, upper_mantle_marker);
  YMIR_GLOBAL_PRODUCTIONF ("gradient of strain exp: %g\n.", inverse_params->grad_strain_exp_proj);

  ymir_vtk_write (mesh, path,
		  upper_mantle_marker, "UM",
 		  grad_strain_exp, "gradient_strain_rate_exponent",
		  NULL);


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);

}




void 
slabs_compute_gradient_yield_stress (slabs_inverse_problem_params_t *inverse_params,
				     const char *vtk_filepath)


{
  const char *this_fn_name = "Computing the gradient of the yield stress.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *euev = inverse_params->euev;
  ymir_dvec_t *viscosity_mod = ymir_vec_clone (inverse_params->viscosity_mod);
  ymir_dvec_t *yielding_marker = inverse_params->yielding_marker;
  ymir_dvec_t *ones = inverse_params->ones;
  ymir_dvec_t *grad_yield_stress = inverse_params->grad_yield_stress;
  double power_sigy = -0.5;
  char                path[BUFSIZ];


  snprintf (path, BUFSIZ, "%s_grad_yield_stress", vtk_filepath);
    
   
  /* apply yielding criteria */
  ymir_dvec_multiply_in (yielding_marker, viscosity_mod);
  ymir_dvec_multiply_in (euev, viscosity_mod);

  /* compute gradient */ 

  ymir_mass_apply (viscosity_mod, grad_yield_stress);
  inverse_params->grad_yield_stress_proj = ymir_vec_innerprod (ones, grad_yield_stress);
  
  YMIR_GLOBAL_PRODUCTIONF ("Computed gradient for the yield stress: %g.\n", inverse_params->grad_yield_stress_proj);

  /* write out vtk file of point-wise gradient of yield stress */
  ymir_vtk_write (mesh, path,
		  grad_yield_stress, "gradient_yield_stress",
		  NULL);
  

 /* Destroy vectors */
  ymir_vec_destroy (viscosity_mod);
#endif
  YMIR_GLOBAL_INFOF ("Finished %s", this_fn_name);

}


void 
slabs_compute_gradient_UM_prefactor (slabs_inverse_problem_params_t *inverse_params,
				       const char *vtk_filepath)

{
  const char *this_fn_name = "Computing the gradient of the upper mantle prefacotr.\n";
#if 1
  ymir_dvec_t *no_yielding_marker = inverse_params->no_yielding_marker;
  ymir_dvec_t *no_bounds_marker = inverse_params->no_bounds_marker;
  ymir_dvec_t *grad_UM_prefactor = inverse_params->grad_UM_prefactor;
  ymir_vec_t *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_dvec_t *euev = inverse_params->euev;
  ymir_dvec_t *viscosity_mod = ymir_vec_clone (inverse_params->viscosity_mod); 
  ymir_dvec_t *ones = inverse_params->ones;
  double power;
  char                path[BUFSIZ];


  ymir_dvec_multiply_in (euev, viscosity_mod);
  ymir_vec_multiply_in1 (upper_mantle_marker, viscosity_mod);
  ymir_mass_apply (viscosity_mod, grad_UM_prefactor);
 
  /* Compute integral */
  inverse_params->grad_upper_mantle_prefactor_proj =  ymir_vec_innerprod (grad_UM_prefactor, ones);
  YMIR_GLOBAL_PRODUCTIONF ("integral 2 is: %g\n.", inverse_params->grad_upper_mantle_prefactor_proj);

  /* Destroy vectors */
  ymir_vec_destroy (viscosity_mod);
#endif 
  YMIR_GLOBAL_INFOF ("Finished %s", this_fn_name);
    
}

void 
slabs_compute_gradient_transition_zone_prefactor (slabs_inverse_problem_params_t *inverse_params,
						  const char *vtk_filepath)


{
  const char *this_fn_name = "Computing the gradient of the transition zone prefactor.\n";
#if 1
  ymir_vec_t *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_dvec_t *euev = inverse_params->euev;
  ymir_dvec_t *visc_temp = ymir_vec_clone (inverse_params->viscosity_mod); 
  ymir_dvec_t *grad_TZ_prefactor = inverse_params->grad_TZ_prefactor;
  ymir_dvec_t *viscosity_mod = ymir_vec_clone (inverse_params->viscosity_mod);
  ymir_dvec_t *ones = inverse_params->ones;
  double power;
  char                path[BUFSIZ];


  ymir_dvec_multiply_in (euev, viscosity_mod);
  ymir_vec_multiply_in1 (transition_zone_marker, viscosity_mod);
  ymir_mass_apply (viscosity_mod, grad_TZ_prefactor);
 
  /* Compute integral */
  inverse_params->grad_transition_zone_prefactor_proj =  ymir_vec_innerprod (grad_TZ_prefactor, ones);
  YMIR_GLOBAL_PRODUCTIONF ("integral 2 is: %g\n.", inverse_params->grad_transition_zone_prefactor_proj);


  /* Destroy vectors */
  ymir_vec_destroy (viscosity_mod);
#endif
  YMIR_GLOBAL_INFOF ("Finished %s", this_fn_name);

}


void 
slabs_compute_gradient_weakzone_prior (slabs_inverse_problem_params_t *inverse_params,
				       const char *vtk_filepath)


{
  const char *this_fn_name = "Computing the gradient of the weakzone prefactor.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *weakzone_marker = inverse_params->weakzone_marker;
  ymir_dvec_t *weakzone_mean = ymir_vec_template (weakzone_marker);
  ymir_dvec_t *weakzone_factor_prior_misfit = inverse_params->weakzone_factor_prior_misfit;
  double       variance_weakfactor = inverse_params->prior_prefactor; 

  char                path[BUFSIZ];

  
    
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
  ymir_vec_add (-1.0, weakzone_marker, weakzone_mean);
  ymir_vec_multiply_in1 (weakzone_marker, weakzone_mean);
  ymir_vec_scale (variance_weakfactor, weakzone_mean);
  
  ymir_mass_apply (weakzone_mean, weakzone_factor_prior_misfit);

  ymir_vtk_write (mesh, path,
		  weakzone_marker, "weakzone",
 		  weakzone_factor_prior_misfit, "weakzone prior misfit",
		  NULL);


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);
  ymir_vec_destroy (weakzone_mean);

}


void
slabs_compute_strain_rate_exponent_prior (slabs_inverse_problem_params_t *inverse_params,
					  const char *vtk_filepath)


{
  const char *this_fn_name = "Computing the gradient of the weakzone prefactor.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_dvec_t *strain_exp_mean = ymir_vec_template (upper_mantle_marker);
  ymir_dvec_t *strain_rate_exponent_prior_misfit = inverse_params->strain_rate_exponent_prior_misfit;
  ymir_dvec_t *strain_exp_distribution = ymir_vec_template (upper_mantle_marker);
  double       variance_strain_exp = inverse_params->prior_strain_exp;
  char                path[BUFSIZ];

  
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
  ymir_vec_scale (inverse_params->strain_exp, strain_exp_distribution);
  ymir_vec_add (-1.0, strain_exp_distribution, strain_exp_mean);
  ymir_vec_multiply_in1 (strain_exp_distribution, strain_exp_mean);
  ymir_vec_scale (variance_strain_exp, strain_exp_mean);
  
  ymir_mass_apply (strain_exp_mean, strain_rate_exponent_prior_misfit);

  ymir_vtk_write (mesh, path,
		  strain_exp_distribution, "weakzone",
 		  strain_rate_exponent_prior_misfit, "weakzone prior misfit",
		  NULL);


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);
  ymir_vec_destroy (strain_exp_mean);
  ymir_vec_destroy (strain_exp_distribution);

}


void
slabs_compute_activation_energy_prior (slabs_inverse_problem_params_t *inverse_params,
				       const char *vtk_filepath)


{
  const char *this_fn_name = "Computing the gradient of the weakzone prefactor.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_dvec_t *activation_energy_mean = ymir_vec_template (upper_mantle_marker);
  ymir_dvec_t *activation_energy_prior_misfit = inverse_params->activation_energy_prior_misfit;
  ymir_dvec_t *activation_energy_distribution = ymir_vec_template (upper_mantle_marker);
  double       variance_activation_energy = inverse_params->prior_activation_energy;
  char                path[BUFSIZ];

  
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
  ymir_vec_scale (inverse_params->strain_exp, activation_energy_distribution);
  ymir_vec_add (-1.0, activation_energy_distribution, activation_energy_mean);
  ymir_vec_multiply_in1 (activation_energy_distribution, activation_energy_mean);
  ymir_vec_scale (variance_activation_energy, activation_energy_mean);
  
  ymir_mass_apply (activation_energy_mean, activation_energy_prior_misfit);

  ymir_vtk_write (mesh, path,
		  activation_energy_distribution, "activation_energy",
 		  activation_energy_prior_misfit, "activation energy misfit",
		  NULL);


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);
  ymir_vec_destroy (activation_energy_mean);
  ymir_vec_destroy (activation_energy_distribution);

}



void
slabs_compute_yield_stress_prior (slabs_inverse_problem_params_t *inverse_params,
				  const char *vtk_filepath)

{
  const char *this_fn_name = "Computing the gradient of the weakzone prefactor.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *ones = inverse_params->ones;
  ymir_dvec_t *yield_stress_mean = ymir_vec_template (ones);
  ymir_dvec_t *yield_stress_prior_misfit = inverse_params->strain_rate_exponent_prior_misfit;
  ymir_dvec_t *yield_stress_distribution = ymir_vec_template (ones);
  double       variance_yield_stress = inverse_params->prior_yield_stress;

  char                path[BUFSIZ];

  
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
  ymir_vec_scale (inverse_params->yield_stress, yield_stress_distribution);
  ymir_vec_add (-1.0, yield_stress_distribution, yield_stress_mean);
  ymir_vec_multiply_in1 (yield_stress_distribution, yield_stress_mean);
  ymir_vec_scale (variance_yield_stress, yield_stress_mean);
  
  ymir_mass_apply (yield_stress_mean, yield_stress_prior_misfit);

  ymir_vtk_write (mesh, path,
		  yield_stress_distribution, "yield stress distribution",
 		  yield_stress_prior_misfit, "yield stress prior misfit",
		  NULL);


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);
  ymir_vec_destroy (yield_stress_mean);
  ymir_vec_destroy (yield_stress_distribution);

}


void
slabs_compute_upper_mantle_prefactor_prior (slabs_inverse_problem_params_t *inverse_params,
					    const char *vtk_filepath)


{
  const char *this_fn_name = "Computing the gradient of the weakzone prefactor.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_dvec_t *upper_mantle_prefactor_mean = ymir_vec_template (upper_mantle_marker);
  ymir_dvec_t *upper_mantle_prefactor_prior_misfit = inverse_params->upper_mantle_prefactor_prior_misfit;
  ymir_dvec_t *upper_mantle_prefactor_distribution = ymir_vec_template (upper_mantle_marker);
  double       variance_upper_mantle_prefactor = inverse_params->prior_upper_mantle_prefactor;

  char                path[BUFSIZ];

  
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
  ymir_vec_scale (inverse_params->upper_mantle_prefactor, upper_mantle_prefactor_distribution);
  ymir_vec_add (-1.0, upper_mantle_prefactor_distribution, upper_mantle_prefactor_mean);
  ymir_vec_multiply_in1 (upper_mantle_prefactor_distribution, upper_mantle_prefactor_mean);
  ymir_vec_scale (variance_upper_mantle_prefactor, upper_mantle_prefactor_mean);
  
  ymir_mass_apply (upper_mantle_prefactor_mean, upper_mantle_prefactor_prior_misfit);
  
  ymir_vtk_write (mesh, path,
		  upper_mantle_prefactor_distribution, "upper mantle",
 		  upper_mantle_prefactor_prior_misfit, "upper mantle prior misfit",
		  NULL);


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);
  ymir_vec_destroy (upper_mantle_prefactor_mean);
  ymir_vec_destroy (upper_mantle_prefactor_distribution);

}


void
slabs_transition_zone_prefactor_prior (slabs_inverse_problem_params_t *inverse_params,
				       const char *vtk_filepath)


{
  const char *this_fn_name = "Computing the gradient of the weakzone prefactor.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_dvec_t *transition_zone_prefactor_mean = ymir_vec_template (transition_zone_marker);
  ymir_dvec_t *transition_zone_prefactor_prior_misfit = inverse_params->transition_zone_prefactor_prior_misfit;
  ymir_dvec_t *transition_zone_prefactor_distribution = ymir_vec_template (transition_zone_marker);
  double       variance_transition_zone_prefactor = inverse_params->prior_transition_zone_prefactor;

  char                path[BUFSIZ];

  
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
  ymir_vec_scale (inverse_params->transition_zone_prefactor, transition_zone_prefactor_distribution);
  ymir_vec_add (-1.0, transition_zone_prefactor_distribution, transition_zone_prefactor_mean);
  ymir_vec_multiply_in1 (transition_zone_prefactor_distribution, transition_zone_prefactor_mean);
  ymir_vec_scale (variance_transition_zone_prefactor, transition_zone_prefactor_mean);
  
  ymir_mass_apply (transition_zone_prefactor_mean, transition_zone_prefactor_prior_misfit);
  
  ymir_vtk_write (mesh, path,
		  transition_zone_prefactor_distribution, "upper mantle",
 		  transition_zone_prefactor_prior_misfit, "upper mantle prior misfit",
		  NULL);


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);
  ymir_vec_destroy (transition_zone_prefactor_mean);
  ymir_vec_destroy (transition_zone_prefactor_distribution);

}


static void
slabs_rhs_viscosity_prior (slabs_inverse_problem_params_t *inverse_params)
{
   ymir_mesh_t          *mesh = inverse_params->mesh;
   ymir_vec_t           *rhs_viscosity_deriv = inverse_params->deriv_viscosity_rhs;
   ymir_vec_t           *rhs_viscosity_prior = inverse_params->viscosity_prior;
   ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
   ymir_vec_t           *u = inverse_params->u;
   ymir_stress_op_t     *stress_op;
   const char           *this_fn_name  = "Constructing RHS of Adjoint";
   
   stress_op  = ymir_stress_op_new (rhs_viscosity_deriv, vel_dir, NULL, u, NULL);
   ymir_stress_op_apply (u, rhs_viscosity_prior, stress_op);
   ymir_stress_op_destroy (stress_op);
 
   YMIR_GLOBAL_PRODUCTIONF ("Into %s\n", this_fn_name);

}

static void
slabs_rhs_viscosity_average (slabs_inverse_problem_params_t *inverse_params)
{
   ymir_mesh_t          *mesh = inverse_params->mesh;
   ymir_vec_t           *rhs_viscosity_deriv = inverse_params->deriv_viscosity_rhs;
   ymir_vec_t           *rhs_viscosity_prior = inverse_params->viscosity_prior;
   ymir_vec_t           *viscosity_average_rhs = inverse_params->viscosity_average_rhs;
   ymir_vel_dir_t       *vel_dir = inverse_params->vel_dir;
   ymir_vec_t           *u = inverse_params->u;
   ymir_stress_op_t     *stress_op;
   const char           *this_fn_name  = "Constructing RHS of Adjoint";
   
   stress_op  = ymir_stress_op_new (viscosity_average_rhs, vel_dir, NULL, u, NULL);
   ymir_stress_op_apply (u, rhs_viscosity_prior, stress_op);
   ymir_stress_op_destroy (stress_op);
   YMIR_GLOBAL_PRODUCTIONF ("Into %s\n", this_fn_name);

}


void 
slabs_compute_gradient_UM_prefactor_viscosity (slabs_inverse_problem_params_t *inverse_params,
					       const char *vtk_filepath)

{
  const char *this_fn_name = "Computing the gradient of the upper mantle prefacotr.\n";
#if 1
  ymir_dvec_t *no_yielding_marker = inverse_params->no_yielding_marker;
  ymir_dvec_t *no_bounds_marker = inverse_params->no_bounds_marker;
  ymir_dvec_t *grad_UM_prefactor = inverse_params->grad_UM_prefactor;
  ymir_vec_t *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_dvec_t *euev = inverse_params->euev;
  ymir_dvec_t *viscosity_mod = ymir_vec_clone (inverse_params->viscosity_mod); 
  ymir_dvec_t *ones = inverse_params->ones;
  double power;
  char                path[BUFSIZ];


  ymir_dvec_multiply_in (euev, viscosity_mod);
  ymir_vec_multiply_in1 (upper_mantle_marker, viscosity_mod);
  ymir_mass_apply (viscosity_mod, grad_UM_prefactor);
 
  /* Compute integral */
  inverse_params->grad_upper_mantle_prefactor_proj =  ymir_vec_innerprod (grad_UM_prefactor, ones);
  YMIR_GLOBAL_PRODUCTIONF ("integral 2 is: %g\n.", inverse_params->grad_upper_mantle_prefactor_proj);

  /* Destroy vectors */
  ymir_vec_destroy (viscosity_mod);
#endif 
  YMIR_GLOBAL_INFOF ("Finished %s", this_fn_name);
    
}




static inline 
void ComputeStressEigenVector(double A[3][3], double Q[3][3], double w[3])
{
  const int n = 3;
  int i, j, p, q, r,  nIter;
  double sd, so;                  // Sums of diagonal resp. off-diagonal elements
  double s, c, t;                 // sin(phi), cos(phi), tan(phi) and temporary storage
  double g, h, z, theta;          // More temporary storage
  double thresh;

  // Initialize Q to the identitity matrix

  for (i=0; i < n; i++)
  {
    Q[i][i] = 1.0;
    for (j=0; j < i; j++)
      Q[i][j] = Q[j][i] = 0.0;
  }


  // Initialize w to diag(A)
  for (i=0; i < n; i++)
    w[i] = A[i][i];

  // Calculate SQR(tr(A))
  sd = 0.0;
  for (i=0; i < n; i++)
    sd += fabs(w[i]);
  sd = sd*sd;

  // Main iteration loop
  for ( nIter=0; nIter < 50; nIter++)
  {
    // Test for convergence
    so = 0.0;
    for (p=0; p < n; p++)
      for (q=p+1; q < n; q++)
        so += fabs(A[p][q]);
    if (so == 0.0)
return;

    if (nIter < 4)
      thresh = 0.2 * so / (n*n);
    else
      thresh = 0.0;

    // Do sweep
    for (p=0; p < n; p++)
      for (q=p+1; q < n; q++)
      {
        g = 100.0 * fabs(A[p][q]);
        if (nIter > 4  &&  fabs(w[p]) + g == fabs(w[p])
                       &&  fabs(w[q]) + g == fabs(w[q]))
        {
          A[p][q] = 0.0;
        }
        else if (fabs(A[p][q]) > thresh)
        {
          // Calculate Jacobi transformation
          h = w[q] - w[p];
          if (fabs(h) + g == fabs(h))
          {
            t = A[p][q] / h;
          }
          else
          {
            theta = 0.5 * h / A[p][q];
            if (theta < 0.0)
              t = -1.0 / (sqrt(1.0 + (theta*theta)) - theta);
            else
              t = 1.0 / (sqrt(1.0 + (theta*theta)) + theta);
          }
          c = 1.0/sqrt(1.0 + (t*t));
          s = t * c;
          z = t * A[p][q];

          // Apply Jacobi transformation
          A[p][q] = 0.0;
          w[p] -= z;
          w[q] += z;
          for (r=0; r < p; r++)
          {
            t = A[r][p];
            A[r][p] = c*t - s*A[r][q];
            A[r][q] = s*t + c*A[r][q];
          }
          for (r=p+1; r < q; r++)
          {
            t = A[p][r];
            A[p][r] = c*t - s*A[r][q];
            A[r][q] = s*t + c*A[r][q];
          }
          for (r=q+1; r < n; r++)
          {
            t = A[p][r];
            A[p][r] = c*t - s*A[q][r];
            A[q][r] = s*t + c*A[q][r];
          }

          // Update eigenvectors
          for (r=0; r < n; r++)
          {
            t = Q[r][p];
            Q[r][p] = c*t - s*Q[r][q];
            Q[r][q] = s*t + c*Q[r][q];
          }
        }
      }
  }
}


void
slabs_stress_tensor_output (ymir_vec_t *stress_tensor,
			    const char *vtk_filepath)
			    /* slabs_inverse_problem_params_t *inverse_params) */
{
  ymir_mesh_t *mesh = stress_tensor->mesh;
  mangll_t    *mangll = mesh->ma;
  ymir_dvec_t *IIe_stress = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_dvec_t *eig_stress = ymir_dvec_new (mesh, 3, YMIR_GAUSS_NODE);
  ymir_dvec_t *eig_col1 = ymir_dvec_new (mesh, 3, YMIR_GAUSS_NODE);
  MPI_Comm            mpicomm = mesh->ma->mpicomm;

  char                path[BUFSIZ];
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *stress_el_mat, *eig_stress_el_mat, *stress_axis_el_mat ;
  double             *x, *y, *z, *tmp_el;
  double              mat[3][3], Q[3][3], w[3];
  mangll_locidx_t     elid;

  /* create work variables */
  stress_el_mat = sc_dmatrix_new (n_nodes_per_el, 6);
  eig_stress_el_mat = sc_dmatrix_new (n_nodes_per_el, 3);
  stress_axis_el_mat = sc_dmatrix_new (n_nodes_per_el, 3);

  x = YMIR_ALLOC (double, n_nodes_per_el);
  y = YMIR_ALLOC (double, n_nodes_per_el);
  z = YMIR_ALLOC (double, n_nodes_per_el);
  tmp_el = YMIR_ALLOC (double, n_nodes_per_el);

  /* sc_dmatrix_set_value (stress_el_mat, 1.0); */



  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    slabs_elem_get_gauss_coordinates (x, y, z, elid, mangll, tmp_el);
    ymir_dvec_get_elem (stress_tensor, stress_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    /* ymir_dvec_get_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, */
    /* 			YMIR_WRITE); */
    ymir_dvec_get_elem (eig_col1, stress_axis_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);


    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict a = stress_el_mat->e[0] + 6 * nodeid;
      double *_sc_restrict b = stress_axis_el_mat->e[0] + 3 * nodeid;

      /* double *_sc_restrict c = stress_el_mat->e[0] + 9 * nodeid; */

      mat[0][0] = a[0];
      mat[0][1] = a[1];
      mat[0][2] = a[2];
      mat[1][0] = a[1];
      mat[1][1] = a[3];
      mat[1][2] = a[4];
      mat[2][0] = a[2];
      mat[2][1] = a[4];
      mat[2][2] = a[5];
      ComputeStressEigenVector(mat, Q, w);
      if ( (w[0]) > (w[1]) && (w[0]) > (w[2])){
	b[0] =  w[0] * Q[0][0];
	b[1] =  w[0] * Q[1][0];
	b[2] =  w[0] * Q[2][0];
      }
      if ((w[0]) > (w[1]) && (w[2]) > (w[0])){
	b[0] =  w[2] * Q[0][2];
	b[1] =  w[2] * Q[1][2];
	b[2] =  w[2] * Q[2][2];

      }
      if  ( (w[1]) > (w[0]) && (w[1]) > (w[2])){
	b[0] =  w[1] * Q[0][1];
	b[1] =  w[1] * Q[1][1];
	b[2] =  w[1] * Q[2][1];
 
      }
      if  ((w[1]) > (w[0]) && (w[2]) > (w[1])){
	b[0] =  w[2] * Q[0][2];
	b[1] =  w[2] * Q[1][2];
	b[2] =  w[2] * Q[2][2];
	
      }
      
	

      YMIR_GLOBAL_PRODUCTIONF ("stuff: %f %f\n", b[1], b[2]);
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (eig_col1, stress_axis_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  ymir_velocity_symtens_dotprod (stress_tensor, stress_tensor, IIe_stress);
  snprintf (path, BUFSIZ, "%s_stress_tensor", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  IIe_stress, "stress",
		  eig_col1, "stress axis",
		  NULL);

  sc_dmatrix_destroy (stress_el_mat);
  sc_dmatrix_destroy (eig_stress_el_mat);
  sc_dmatrix_destroy (stress_axis_el_mat);
  YMIR_FREE (x);
  YMIR_FREE (y);
  YMIR_FREE (z);
  YMIR_FREE (tmp_el);
  ymir_vec_destroy (IIe_stress);
  ymir_vec_destroy (eig_stress);
  ymir_vec_destroy (eig_col1);

}

