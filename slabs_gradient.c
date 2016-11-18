  
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
  inverse_problem_grad->adjoint_vq = NULL;
  /* inverse_problem_grad->surface_velocity_misfit = NULL; */
  /* inverse_problem_grad->surface_normal_stress_misfit = NULL; */
  /* inverse_problem_grad->surface_IIe_misfit = NULL; */
  
  /* assign pointers to gradient vectors */
  inverse_problem_grad->grad_weak_factor = NULL; 
  inverse_problem_grad->grad_strain_exp = NULL;
  inverse_problem_grad->grad_yield_stress = NULL;
  inverse_problem_grad->grad_TZ_prefactor = NULL;
  inverse_problem_grad->grad_UM_prefactor = NULL;
  inverse_problem_grad->grad_activation_energy = NULL;
  /* assign pointers to incremental update directions */
  inverse_problem_grad->inc_update_strain_rate_exp_vec = NULL;
  inverse_problem_grad->inc_update_yield_stress_vec = NULL;
  inverse_problem_grad->inc_update_UM_prefactor_vec = NULL;
  inverse_problem_grad->inc_update_TZ_prefactor_vec = NULL;
  inverse_problem_grad->inc_update_activation_energy_vec = NULL;
  inverse_problem_grad->inc_update_weakzone_prefactor_vec = NULL;

  inverse_problem_grad->sam_weakzone_stencil = NULL;
  inverse_problem_grad->izu_weakzone_stencil = NULL;
  inverse_problem_grad->ryu_weakzone_stencil = NULL;



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
slabs_clear_inverse_problem_empty (slabs_inverse_problem_params_t *inverse_params)
{
  YMIR_FREE (inverse_params);

}

void
slabs_initialize_weakzone_stencils (slabs_inverse_problem_params_t *inverse_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;

  const char    *this_fn_name = "creating new inverse problem";     
  inverse_params->sam_weakzone_stencil = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->izu_weakzone_stencil = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->ryu_weakzone_stencil = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  
}


void
slabs_destroy_weakzone_stencils (slabs_inverse_problem_params_t *inverse_params)
{
  ymir_vec_destroy (inverse_params->sam_weakzone_stencil);
  ymir_vec_destroy (inverse_params->izu_weakzone_stencil);
  ymir_vec_destroy (inverse_params->ryu_weakzone_stencil);
}


void
slabs_initialize_gradients_velocity_data (slabs_inverse_problem_params_t *inverse_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;

  const char    *this_fn_name = "creating new inverse problem";     
  /* assign pointers to gradient vectors */
  inverse_params->grad_weak_factor = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE); 
  inverse_params->grad_strain_exp = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->grad_yield_stress = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->grad_TZ_prefactor = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->grad_UM_prefactor = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->grad_activation_energy = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
 
  
}




void
slabs_destroy_velocity_gradients (slabs_inverse_problem_params_t *inverse_params)
{
  const char    *this_fn_name = "destroying gradient vectors";     

  ymir_vec_destroy (inverse_params->grad_weak_factor);
  ymir_vec_destroy (inverse_params->grad_strain_exp);
  ymir_vec_destroy (inverse_params->grad_yield_stress);
  ymir_vec_destroy (inverse_params->grad_TZ_prefactor);
  ymir_vec_destroy (inverse_params->grad_UM_prefactor);
  ymir_vec_destroy (inverse_params->grad_activation_energy);

  YMIR_GLOBAL_INFOF ("Finished %s\n", this_fn_name);

}


void
slabs_initialize_gradients_viscosity_data (slabs_inverse_problem_params_t *inverse_params)
{
  ymir_mesh_t *mesh = inverse_params->mesh;

  const char    *this_fn_name = "creating new inverse problem";     
  /* assign pointers to gradient vectors */
  inverse_params->grad_viscosity_weakfactor = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE); 
  inverse_params->grad_viscosity_strain_rate_exponent = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->grad_viscosity_yield_stress = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->grad_viscosity_TZ_prefactor = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->grad_viscosity_UM_prefactor = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  inverse_params->grad_viscosity_activation_energy = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
 
  
}


void
slabs_destroy_gradients_viscosity_data (slabs_inverse_problem_params_t *inverse_params)
{
  const char    *this_fn_name = "destroying gradient vectors";     

  ymir_vec_destroy (inverse_params->grad_viscosity_weakfactor);
  ymir_vec_destroy (inverse_params->grad_viscosity_strain_rate_exponent);
  ymir_vec_destroy (inverse_params->grad_viscosity_yield_stress);
  ymir_vec_destroy (inverse_params->grad_viscosity_TZ_prefactor);
  ymir_vec_destroy (inverse_params->grad_viscosity_UM_prefactor);
  ymir_vec_destroy (inverse_params->grad_viscosity_activation_energy);

  YMIR_GLOBAL_INFOF ("Finished %s\n", this_fn_name);

}





void
slabs_initialize_inverse_params (slabs_inverse_problem_params_t *inverse_params,
				      slabs_nl_stokes_problem_t *nl_stokes,
				      slabs_nl_solver_options_t *solver_options,
				      slabs_physics_options_t *physics_options)
{
  const char *this_fn_name = "Initializing inverse params data structures.\n";

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


/**
 * Initializes the weak zone.
 */
void
slabs_setup_mesh_data_weakzone (slabs_inverse_problem_params_t *inverse_params,
				slabs_stokes_state_t *state,
				slabs_physics_options_t *physics_options,
				const char *vtk_filepath)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t *sam_weak =  inverse_params->sam_weakzone_stencil;
  ymir_vec_t *izu_weak =  inverse_params->izu_weakzone_stencil;
  ymir_vec_t *ryu_weak =  inverse_params->ryu_weakzone_stencil;
  MPI_Comm            mpicomm = mesh->ma->mpicomm;
  char                path[BUFSIZ];
    


  /* initialization of weak zone computation */
  slabs_physics_init_sam_data_weakzone (mpicomm, physics_options);
  slabs_physics_init_izu_data_weakzone (mpicomm, physics_options);
  slabs_physics_init_ryu_data_weakzone (mpicomm, physics_options);

#if 1
  /* fill Stokes state with weak zone factor values */
  slabs_physics_compute_sam_weakzone (sam_weak, physics_options);
  slabs_physics_compute_izu_weakzone (izu_weak, physics_options);
  slabs_physics_compute_ryu_weakzone (ryu_weak, physics_options);

#endif
  /* Write out stencil to vtk file */
  snprintf (path, BUFSIZ, "%s_sam_weak", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  sam_weak, "sam weakzone",
		  izu_weak, "izu weakzone",
		  ryu_weak, "ryu weakzone",
		  NULL);
  
  slabs_physics_data_clear_weakzone (physics_options);

  

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


void
slabs_inverse_problem_destroy (slabs_inverse_problem_params_t *inverse_params)
{
  const char    *this_fn_name = "destroy data";

  slabs_inverse_problem_clear (inverse_params);
  YMIR_FREE (inverse_params);
  YMIR_GLOBAL_INFOF ("Done %s\n", this_fn_name);

}


void
slabs_gradient_viscosity_param_weakfactor (slabs_inverse_problem_params_t *inverse_params,
					   const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t           *weakfactor_stencil = inverse_params->weakzone_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat;
  sc_dmatrix_t       *weakfactor_stencil_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  weakfactor_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
      
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (weakfactor_stencil, weakfactor_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_deriv = grad_viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict weakfactor_distrib = weakfactor_stencil_el_mat->e[0] +  nodeid;
 
      viscosity_deriv[0] = weakfactor_distrib[0] * visc[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_weakfactor, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
  }

  snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  grad_viscosity_weakfactor, "gradient of viscosity w.r.t weakfactor",
		  NULL);

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (weakfactor_stencil_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);
}

void
slabs_gradient_viscosity_param_strain_rate_exponent (slabs_inverse_problem_params_t *inverse_params,
						     const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_strain_rate_exponent = inverse_params->grad_viscosity_strain_rate_exponent;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat;
  sc_dmatrix_t       *upper_mantle_stencil_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_strain_rate_exponent, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_deriv = grad_viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict strain_exp_distrib = upper_mantle_stencil_el_mat->e[0] +  nodeid;
 
      viscosity_deriv[0] = strain_exp_distrib[0] * visc[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_strain_rate_exponent, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
  }

  snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  grad_viscosity_strain_rate_exponent, "gradient of viscosity w.r.t strain_exp",
		  NULL);

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (upper_mantle_stencil_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);

}


void
slabs_gradient_viscosity_param_upper_mantle (slabs_inverse_problem_params_t *inverse_params,
					     const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_upper_mantle = inverse_params->grad_viscosity_UM_prefactor;
  ymir_vec_t           *upper_mantle_marker = inverse_params->upper_mantle_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat;
  sc_dmatrix_t       *upper_mantle_stencil_el_mat;
  double             *x, *y, *z, *tmp_el;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_upper_mantle, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_deriv = grad_viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict upper_mantle = upper_mantle_stencil_el_mat->e[0] +  nodeid;
 
      viscosity_deriv[0] = upper_mantle[0] * visc[0];      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_upper_mantle, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
  }

  snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  grad_viscosity_upper_mantle, "gradient of viscosity w.r.t upper_mantle",
		  NULL);


  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (upper_mantle_stencil_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);
}


void
slabs_gradient_viscosity_param_transition_zone (slabs_inverse_problem_params_t *inverse_params,
						const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_transition_zone = inverse_params->grad_viscosity_TZ_prefactor;
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
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  transition_zone_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
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

}


void
slabs_gradient_viscosity_param_yield_stress (slabs_inverse_problem_params_t *inverse_params,
					     const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_yield_stress = inverse_params->grad_yield_stress;
  ymir_vec_t           *yielding_marker = inverse_params->yielding_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat;
  sc_dmatrix_t       *yielding_el_mat;
  mangll_locidx_t     elid;

  
  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  yielding_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (yielding_marker, yielding_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_yield_stress, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_deriv = grad_viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict yield = yielding_el_mat->e[0] +  nodeid;
 
      viscosity_deriv[0] = yield[0] * visc[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_yield_stress, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  grad_viscosity_yield_stress, "gradient of viscosity w.r.t yield stress",
		  NULL);



  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (yielding_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);

}

void 
slabs_compute_gradient_strain_rate_exponent_average_viscosity (slabs_inverse_problem_params_t *inverse_params,
							       const char *vtk_filepath)
{
  const char *this_fn_name = "Computing the gradient of the strain rate exponent.\n";
  const double grad_strain_exp_proj = inverse_params->grad_strain_exp_proj;
  const double average_visc_area = inverse_params->average_visc_area;

#if 1
  inverse_params->grad_strain_exp_visc_avg = grad_strain_exp_proj * average_visc_area;


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);

}

void 
slabs_compute_gradient_weakfactor_average_viscosity (slabs_inverse_problem_params_t *inverse_params,
							       const char *vtk_filepath)
{
  const char *this_fn_name = "Computing the gradient of the strain rate exponent.\n";
  const double grad_weak_factor_proj = inverse_params->grad_weak_factor_proj;
  const double average_visc_area = inverse_params->average_visc_area;

#if 1
  inverse_params->grad_weak_factor_visc_avg = grad_weak_factor_proj * average_visc_area;


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);

}
void 
slabs_compute_gradient_yield_stress_average_viscosity (slabs_inverse_problem_params_t *inverse_params,
						       const char *vtk_filepath)
{
  const char *this_fn_name = "Computing the gradient of the strain rate exponent.\n";
  const double grad_yield_stress_proj = inverse_params->grad_yield_stress_proj;
  const double average_visc_area = inverse_params->average_visc_area;

#if 1
  inverse_params->grad_yield_stress_visc_avg = grad_yield_stress_proj * average_visc_area;


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);

}
void 
slabs_compute_gradient_activation_energy_average_viscosity (slabs_inverse_problem_params_t *inverse_params,
							    const char *vtk_filepath)
{
  const char *this_fn_name = "Computing the gradient of the strain rate exponent.\n";
  const double grad_activation_energy_proj = inverse_params->grad_activation_energy_proj;
  const double average_visc_area = inverse_params->average_visc_area;

#if 1
  inverse_params->grad_activation_energy_visc_avg = grad_activation_energy_proj * average_visc_area;


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);

}
void 
slabs_compute_gradient_upper_mantle_average_viscosity (slabs_inverse_problem_params_t *inverse_params,
						       const char *vtk_filepath)
{
  const char *this_fn_name = "Computing the gradient of the strain rate exponent.\n";
  const double grad_upper_mantle_proj = inverse_params->grad_upper_mantle_prefactor_proj;
  const double average_visc_area = inverse_params->average_visc_area;

#if 1
  inverse_params->grad_upper_mantle_prefactor_visc_avg = grad_upper_mantle_proj * average_visc_area;


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);

}
void 
slabs_compute_gradient_transition_zone_average_viscosity (slabs_inverse_problem_params_t *inverse_params,
							  const char *vtk_filepath)
{
  const char *this_fn_name = "Computing the gradient of the strain rate exponent.\n";
  const double grad_transition_zone_proj = inverse_params->grad_transition_zone_prefactor_proj;
  const double average_visc_area = inverse_params->average_visc_area;

#if 1
  inverse_params->grad_transition_zone_prefactor_visc_avg = grad_transition_zone_proj * average_visc_area;


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);

}
void 
slabs_compute_gradient_lower_mantle_average_viscosity (slabs_inverse_problem_params_t *inverse_params,
						       const char *vtk_filepath)
{
  const char *this_fn_name = "Computing the gradient of the strain rate exponent.\n";
  const double grad_lower_mantle_proj = inverse_params->grad_lower_mantle_prefactor_proj;
  const double average_visc_area = inverse_params->average_visc_area;

#if 1
  inverse_params->grad_lower_mantle_prefactor_visc_avg = grad_lower_mantle_proj * average_visc_area;


  /* Destroy vectors */
#endif
  YMIR_GLOBAL_INFOF ("Done %s", this_fn_name);

}



void 
slabs_compute_stress_second_invariant (slabs_inverse_problem_params_t *inverse_params,
				       const char *vtk_filepath)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t *eu = inverse_params->eu;
  ymir_vec_t *viscosity = inverse_params->viscosity;
  ymir_vec_t *eu_dummy = ymir_vec_template (eu);
  ymir_vec_t *second_invariant_stress = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);

#if 1
  ymir_vec_multiply_in1 (viscosity, eu);
  ymir_velocity_symtens_dotprod (eu_dummy, eu_dummy, second_invariant_stress);

  ymir_vec_destroy (eu_dummy);
  ymir_vec_destroy (second_invariant_stress);
#endif
}


#if 1
void 
slabs_compute_stress_second_invariant_adjoint_rhs (slabs_inverse_problem_params_t *inverse_params,
						   const char *vtk_filepath)
{
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_vec_t *eu = inverse_params->eu;
  ymir_vec_t *viscosity = inverse_params->viscosity;
  ymir_vec_t *eu_dummy = ymir_vec_clone (eu);
  ymir_vec_t *second_invariant_stress1 = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_vec_t *second_invariant_stress2 = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_vec_t *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t *u = inverse_params->u;
  ymir_vec_t *rhs1 = ymir_vec_template (u);
  ymir_vec_t *rhs_stress_second_invariant = ymir_vec_template (u);
  ymir_vel_dir_t *vel_dir = inverse_params->vel_dir;
  ymir_stress_op_t     *stress_op1, *stress_op2;
  const char           *this_fn_name  = "Constructing RHS of Adjoint";


  ymir_vec_multiply_in1 (viscosity, eu_dummy);
  ymir_vec_scale (2.0, eu_dummy);
  /* Part: div (8*viscosity^2 * e(u)) */
  ymir_vec_multiply_in1 (viscosity, second_invariant_stress1);
  ymir_vec_scale (8.0, second_invariant_stress1);
  /* Part  : 8*viscosity*viscosity,IIe*(eu:eu)*(eu:eu_test) */

  ymir_velocity_symtens_dotprod (eu_dummy, eu_dummy, second_invariant_stress2);
  ymir_vec_multiply_in1 (viscosity, second_invariant_stress2);
  ymir_vec_multiply_in1 (viscosity_deriv_IIe, second_invariant_stress2);
  ymir_vec_scale (8.0, second_invariant_stress2);
  stress_op1  = ymir_stress_op_new ( second_invariant_stress1, vel_dir, NULL, u, NULL);
  stress_op2  = ymir_stress_op_new ( second_invariant_stress2, vel_dir, NULL, u, NULL);
  
  ymir_stress_op_apply (u, rhs1, stress_op1);
  ymir_stress_op_apply (u, rhs_stress_second_invariant, stress_op2);
  ymir_vec_add (1.0, rhs1, rhs_stress_second_invariant);

  ymir_stress_op_destroy (stress_op1);
  ymir_stress_op_destroy (stress_op2);


  ymir_vec_destroy (eu_dummy);

}

#endif




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
slabs_compute_gradient_activation_energy (slabs_inverse_problem_params_t *inverse_params,
					  const char *vtk_filepath)


{
  const char *this_fn_name = "Computing the gradient of the activation energy.\n";
#if 0 
  ymir_dvec_t *no_yielding = inverse_params->no_yielding;
  ymir_dvec_t *no_upper_bound = inverse_params->no_upper_bound;
  ymir_dvec_t *grad_activation_energy = inverse_params->grad_activation_energy;
  ymir_dvec_t *euev = ymir_dvec_new (mesh, 1, YMIR_GAUSS_NODE);
  ymir_dvec_t *viscosity_mod = ymir_vec_clone (inverse_params->viscosity_mod);
  ymir_dvec_t *ones = inverse_params->ones;
  double  e_a = physics_options->viscosity_temp_decay;
  double power;
  char                path[BUFSIZ];

 

  ymir_dvec_multiply_in (euev, viscosity_mod);  
  ymir_dvec_t *deriv = ymir_vec_template (state->weak_vec);
  slabs_physics_viscosity_deriv_activation_energy (deriv, state, physics_options);
  ymir_vec_multiply_in1 (viscosity_mod, deriv);
  ymir_mass_apply (deriv, grad_activation_energy);

  
  /* Compute integral */
  double grad_active_energy = inverse_params->grad_activ_energy;
  grad_active_energy =  ymir_vec_innerprod (grad_activation_energy, ones);
  YMIR_GLOBAL_PRODUCTIONF ("integral 2 is: %g\n.", grad_active_energy);


  /* Destroy vectors */
  ymir_vec_destroy (viscosity_mod);
  ymir_vec_destroy (deriv);
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
  double       variance_weakfactor = inverse_params->prior_weakfactor_scale; 

  char                path[BUFSIZ];

  
    
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
  ymir_vec_scale (inverse_params->prior_weakfactor_scale, weakzone_mean);
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
  double       variance_strain_exp = inverse_params->prior_strain_rate_exponent_scale;
  char                path[BUFSIZ];

  
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
  ymir_vec_scale (inverse_params->prior_strain_rate_exponent_scale, strain_exp_mean);  
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
  ymir_vec_scale (inverse_params->prior_activation_energy_scale, activation_energy_mean);  
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
  ymir_vec_scale (inverse_params->prior_yield_stress_scale, yield_stress_mean);  
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
  ymir_vec_scale (inverse_params->prior_upper_mantle_prefactor_scale, upper_mantle_prefactor_mean);  
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
  ymir_vec_scale (inverse_params->prior_transition_zone_prefactor_scale, transition_zone_prefactor_mean);  
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
slabs_average_viscosity_area (slabs_nl_stokes_problem_t *nl_stokes,
			      slabs_inverse_problem_params_t *inverse_params)
{
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *log_viscosity = inverse_params->log_viscosity;
  ymir_vec_t           *ones = inverse_params->ones;
  ymir_vec_t           *out = ymir_vec_clone (viscosity_stencil);

  ymir_mass_apply (ones, out);
  inverse_params->average_visc_area = exp (ymir_vec_innerprod (log_viscosity, out)) ;
  inverse_params->average_visc_area_misfit_cost = 0.5 * (inverse_params->average_visc_area - inverse_params->average_viscosity_data)
    * (inverse_params->average_visc_area - inverse_params->average_viscosity_data);
  inverse_params->average_visc_area_misfit = 0.5 * (inverse_params->average_visc_area - inverse_params->average_viscosity_data);
  
}
 
#if 1
void 
slabs_gradient_viscosity_average (slabs_nl_stokes_problem_t *nl_stokes,
				  slabs_inverse_problem_params_t *inverse_params,
				  const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = nl_stokes->viscosity;
  ymir_vec_t           *inverse_viscosity = ymir_vec_template (viscosity);
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_param = inverse_params->deriv_viscosity_param;
  ymir_vec_t           *out = inverse_params->deriv_viscosity_rhs;
  ymir_vec_t           *grad_visc = ymir_vec_template (viscosity);
  ymir_vec_t           *ones = inverse_params->ones;
  double               average_visc_area = inverse_params->average_visc_area;
  double               gradient_visc_area = inverse_params->gradient_visc_area;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *inverse_viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *deriv_viscosity_el_mat, *out_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  inverse_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  deriv_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_param, deriv_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (inverse_viscosity, inverse_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);


    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict viscosity_area = viscosity_stencil_el_mat->e[0] +  nodeid;
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_deriv = deriv_viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = -viscosity_area[0] * viscosity_deriv[0] / visc[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }
  ymir_vec_scale (average_visc_area, out);
  ymir_mass_apply (out, grad_visc);
  inverse_params->gradient_visc_area = ymir_vec_innerprod (grad_visc, ones);

  snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  inverse_viscosity, "inverse viscosity",
		  NULL);



  sc_dmatrix_destroy (inverse_viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (deriv_viscosity_el_mat);
  sc_dmatrix_destroy (out_el_mat);
  ymir_vec_destroy (inverse_viscosity);
  ymir_vec_destroy (grad_visc);

}

#endif


void 
slabs_viscosity_average_velocity_deriv (slabs_nl_stokes_problem_t *nl_stokes,
					slabs_inverse_problem_params_t *inverse_params,
					const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = nl_stokes->viscosity;
  ymir_vec_t           *inverse_viscosity = ymir_vec_template (viscosity);
  ymir_vec_t           *viscosity_stencil = inverse_params->viscosity_stencil;
  ymir_vec_t           *viscosity_deriv_param = inverse_params->deriv_viscosity_param;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *out = inverse_params->viscosity_average_rhs;
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
  sc_dmatrix_t       *viscosity_el_mat, *inverse_viscosity_el_mat;
  sc_dmatrix_t       *viscosity_stencil_el_mat, *viscosity_deriv_el_mat, *out_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  inverse_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  viscosity_deriv_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  out_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (inverse_viscosity, inverse_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);
    ymir_dvec_get_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);


    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_deriv = viscosity_deriv_el_mat->e[0] +  nodeid;
      double *_sc_restrict output = out_el_mat->e[0] +  nodeid;
 
      output[0] = average_visc_area_misfit * average_visc_area_misfit * viscosity_deriv[0] / visc[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (out, out_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);

  }

  snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  inverse_viscosity, "inverse viscosity",
		  NULL);



  sc_dmatrix_destroy (inverse_viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_stencil_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_el_mat);
  sc_dmatrix_destroy (out_el_mat);
  ymir_vec_destroy (inverse_viscosity);
}



void
slabs_log_viscosity ( slabs_nl_stokes_problem_t *nl_stokes,
		     const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = nl_stokes->viscosity;
  ymir_vec_t           *inverse_viscosity = ymir_vec_template (viscosity);
  ymir_vec_t           *log_viscosity = ymir_vec_template (viscosity);
  ymir_vec_t           *viscosity_IIe = ymir_vec_clone (nl_stokes->stokes_op->stress_op->dvdIIe);
  ymir_vec_t           *IIe = ymir_vec_template (viscosity);
  ymir_vec_t           *u = nl_stokes->stokes_op->stress_op->usol;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  ymir_velocity_elem_t *vel_elem = ymir_velocity_elem_new (mesh->cnodes->N, mesh->ma->ompsize);
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *inverse_viscosity_el_mat;
  sc_dmatrix_t       *log_viscosity_el_mat;
  double             *x, *y, *z, *tmp_el;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  inverse_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  log_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
  x = YMIR_ALLOC (double, n_nodes_per_el);
  y = YMIR_ALLOC (double, n_nodes_per_el);
  z = YMIR_ALLOC (double, n_nodes_per_el);
  tmp_el = YMIR_ALLOC (double, n_nodes_per_el);
  ymir_second_invariant_vec (u, IIe, vel_elem);
     

 
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    slabs_elem_get_gauss_coordinates (x, y, z, elid, mangll, tmp_el);
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    /* ymir_dvec_get_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, */
    /* 			YMIR_WRITE); */
    ymir_dvec_get_elem (inverse_viscosity, inverse_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);
    ymir_dvec_get_elem (log_viscosity, inverse_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);


    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict inverse_visc = inverse_viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict log_visc = log_viscosity_el_mat->e[0] +  nodeid;

      inverse_visc[0] = 1. / visc[0];
      log_visc[0] = log (visc[0]);

      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (inverse_viscosity, inverse_viscosity_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);


  }

  snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath);
  ymir_vtk_write (mesh, path,
		  inverse_viscosity, "inverse viscosity",
		  log_viscosity, "log viscosity",
		  NULL);



  sc_dmatrix_destroy (inverse_viscosity_el_mat);
  sc_dmatrix_destroy (log_viscosity_el_mat);
  sc_dmatrix_destroy (viscosity_el_mat);
  ymir_vec_destroy (log_viscosity);
  ymir_vec_destroy (inverse_viscosity);
  ymir_vec_destroy (viscosity_IIe);
  ymir_velocity_elem_destroy (vel_elem);
  ymir_vec_destroy (IIe);
  YMIR_FREE (x);
  YMIR_FREE (y);
  YMIR_FREE (z);
  YMIR_FREE (tmp_el);
		    
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




