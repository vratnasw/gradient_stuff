  
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
  inverse_problem_grad->adjoint_vq = NULL;//ymir_vec_template (state->vel_press_vec);
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


void
slabs_second_deriv_viscosity_IIe_param_weakfactor (slabs_inverse_problem_params_t *inverse_params,
						   const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_weakfactor = inverse_params->grad_viscosity_weakfactor;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_weakfactor = inverse_params->grad_viscosity_second_deriv_IIe_weakfactor;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *weakfactor_stencil = inverse_params->weakzone_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat, *viscosity_deriv_IIe_el_mat;
  sc_dmatrix_t       *weakfactor_stencil_el_mat, *grad_viscosity_second_deriv_IIe_weakfactor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  weakfactor_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_weakfactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe = sc_dmatrix_new (n_nodes_per_el, 1);
      
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (weakfactor_stencil, weakfactor_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_weakfactor, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_weakfactor, grad_viscosity_second_deriv_IIe_weakfactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict weakfactor_distrib = weakfactor_stencil_el_mat->e[0] +  nodeid;
      double *_sc_restrict second_deriv_IIe_weak = grad_viscosity_second_deriv_IIe_weakfactor_el_mat->e[0] + nodeid;

      second_deriv_IIe_weak[0] = weakfactor_distrib[0] * viscosity_IIe_deriv[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_second_deriv_IIe_weakfactor, grad_viscosity_second_deriv_IIe_weakfactor_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  grad_viscosity_weakfactor, "gradient of viscosity w.r.t weakfactor", */
  /* 		  NULL); */

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (weakfactor_stencil_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_weakfactor_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);

}

void
slabs_second_deriv_viscosity_IIe_param_strain_rate_exponent (slabs_inverse_problem_params_t *inverse_params,
							     const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_strain_rate_exponent = inverse_params->grad_viscosity_second_deriv_IIe_strain_rate_exponent;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *upper_mantle_stencil = inverse_params->upper_mantle_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat, *viscosity_deriv_IIe_el_mat;
  sc_dmatrix_t       *upper_mantle_stencil_el_mat, *grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe = sc_dmatrix_new (n_nodes_per_el, 1);
      
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_stencil, upper_mantle_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_strain_rate_exponent, grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict upper_mantle_distrib = upper_mantle_stencil_el_mat->e[0] +  nodeid;
      double *_sc_restrict second_deriv_IIe_weak = grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat->e[0] + nodeid;

      second_deriv_IIe_weak[0] = upper_mantle_distrib[0] * viscosity_IIe_deriv[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_second_deriv_IIe_strain_rate_exponent, grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  grad_viscosity_weakfactor, "gradient of viscosity w.r.t weakfactor", */
  /* 		  NULL); */

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (upper_mantle_stencil_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_strain_rate_exponent_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);

}

void
slabs_second_deriv_viscosity_IIe_param_yield_stress (slabs_inverse_problem_params_t *inverse_params,
							     const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_yield_stress = inverse_params->grad_viscosity_second_deriv_IIe_yield_stress;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *yielding_marker_stencil = inverse_params->yielding_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat, *viscosity_deriv_IIe_el_mat;
  sc_dmatrix_t       *yielding_marker_stencil_el_mat, *grad_viscosity_second_deriv_IIe_yield_stress_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  yielding_marker_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_yield_stress_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe = sc_dmatrix_new (n_nodes_per_el, 1);
      
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (yielding_marker_stencil, yielding_marker_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_yield_stress, grad_viscosity_second_deriv_IIe_yield_stress_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict yield_points = yielding_marker_stencil_el_mat->e[0] +  nodeid;
      double *_sc_restrict second_deriv_IIe_yield = grad_viscosity_second_deriv_IIe_yield_stress_el_mat->e[0] + nodeid;

      second_deriv_IIe_yield[0] = yield_points[0] * viscosity_IIe_deriv[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_second_deriv_IIe_yield_stress, grad_viscosity_second_deriv_IIe_yield_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  grad_viscosity_weakfactor, "gradient of viscosity w.r.t weakfactor", */
  /* 		  NULL); */

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (yielding_marker_stencil_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_yield_stress_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);

}

void
slabs_second_deriv_viscosity_IIe_param_upper_mantle_prefactor (slabs_inverse_problem_params_t *inverse_params,
							       const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_UM_prefactor = inverse_params->grad_viscosity_second_deriv_IIe_UM_prefactor;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *upper_mantle_stencil = inverse_params->upper_mantle_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat, *viscosity_deriv_IIe_el_mat;
  sc_dmatrix_t       *upper_mantle_stencil_el_mat, *grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe = sc_dmatrix_new (n_nodes_per_el, 1);
      
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_stencil, upper_mantle_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_UM_prefactor, grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict upper_mantle = upper_mantle_stencil_el_mat->e[0] +  nodeid;
      double *_sc_restrict second_deriv_IIe_UM = grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat->e[0] + nodeid;

      second_deriv_IIe_UM[0] = upper_mantle[0] * viscosity_IIe_deriv[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_second_deriv_IIe_UM_prefactor, grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  grad_viscosity_weakfactor, "gradient of viscosity w.r.t weakfactor", */
  /* 		  NULL); */

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (upper_mantle_stencil_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_UM_prefactor_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);

}


void
slabs_second_deriv_viscosity_IIe_param_transition_zone_prefactor (slabs_inverse_problem_params_t *inverse_params,
								  const char *vtk_filepath)
{
  ymir_vec_t           *viscosity = inverse_params->viscosity;
  ymir_vec_t           *grad_viscosity_second_deriv_IIe_TZ_prefactor = inverse_params->grad_viscosity_second_deriv_IIe_TZ_prefactor;
  ymir_vec_t           *viscosity_deriv_IIe = inverse_params->viscosity_deriv_IIe;
  ymir_vec_t           *transition_zone_stencil = inverse_params->transition_zone_marker;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat, *viscosity_deriv_IIe_el_mat;
  sc_dmatrix_t       *transition_zone_stencil_el_mat, *grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  transition_zone_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  viscosity_deriv_IIe = sc_dmatrix_new (n_nodes_per_el, 1);
      
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (transition_zone_stencil, transition_zone_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (viscosity_deriv_IIe, viscosity_deriv_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);
    ymir_dvec_get_elem (grad_viscosity_second_deriv_IIe_TZ_prefactor, grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_IIe_deriv = viscosity_deriv_IIe_el_mat->e[0] + nodeid;
      double *_sc_restrict transition_zone = transition_zone_stencil_el_mat->e[0] +  nodeid;
      double *_sc_restrict second_deriv_IIe_TZ = grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat->e[0] + nodeid;

      second_deriv_IIe_TZ[0] = transition_zone[0] * viscosity_IIe_deriv[0];
      
    }
     /* ymir_dvec_set_elem (eig_stress, eig_stress_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET); */
     ymir_dvec_set_elem (grad_viscosity_second_deriv_IIe_TZ_prefactor, grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat, YMIR_STRIDE_NODE, elid, YMIR_SET);
  }

  /* snprintf (path, BUFSIZ, "%s_log_viscosity", vtk_filepath); */
  /* ymir_vtk_write (mesh, path, */
  /* 		  grad_viscosity_weakfactor, "gradient of viscosity w.r.t weakfactor", */
  /* 		  NULL); */

  sc_dmatrix_destroy (viscosity_el_mat);
  sc_dmatrix_destroy (transition_zone_stencil_el_mat);
  sc_dmatrix_destroy (grad_viscosity_el_mat);
  sc_dmatrix_destroy (grad_viscosity_second_deriv_IIe_TZ_prefactor_el_mat);
  sc_dmatrix_destroy (viscosity_deriv_IIe_el_mat);

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
  ymir_dvec_t *log_IIe = inverse_params->log_IIe;
  ymir_mesh_t          *mesh = viscosity->mesh;
  mangll_t             *mangll = mesh->ma;
  char                path[BUFSIZ];
  double n = inverse_params->strain_exp;
  double log_factor = -0.5 / (n);
  double power = (1 - n)/ (n + n);
  
  const mangll_locidx_t  n_elements = mesh->cnodes->K;
  const unsigned int  N = ymir_n (mangll->N);
  const unsigned int  n_nodes_per_el = (N + 1) * (N + 1) * (N + 1);
  int nodeid;
  sc_dmatrix_t       *viscosity_el_mat, *grad_viscosity_el_mat;
  sc_dmatrix_t       *upper_mantle_stencil_el_mat, *log_IIe_el_mat;
  mangll_locidx_t     elid;


  /* create work variables */
  viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  upper_mantle_stencil_el_mat = sc_dmatrix_new (n_nodes_per_el, 1); 
  grad_viscosity_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  log_IIe_el_mat = sc_dmatrix_new (n_nodes_per_el, 1);
  
  
  for (elid = 0; elid < n_elements; elid++) { /* loop over all elements */
    /* get coordinates of this element at Gauss nodes */
    ymir_dvec_get_elem (viscosity, viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (upper_mantle_marker, upper_mantle_stencil_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (log_IIe, log_IIe_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_READ);
    ymir_dvec_get_elem (grad_viscosity_strain_rate_exponent, grad_viscosity_el_mat, YMIR_STRIDE_NODE, elid,
			YMIR_WRITE);

    for (nodeid = 0; nodeid < n_nodes_per_el; nodeid++){
      double *_sc_restrict visc = viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict viscosity_deriv = grad_viscosity_el_mat->e[0] +  nodeid;
      double *_sc_restrict strain_exp_distrib = upper_mantle_stencil_el_mat->e[0] +  nodeid;
      double *_sc_restrict logIIe = log_IIe_el_mat->e[0] + nodeid;
      viscosity_deriv[0] = logIIe[0] * log_factor * strain_exp_distrib[0] * visc[0];
      
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
  sc_dmatrix_destroy (log_IIe_el_mat);


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




#if 0
static void 
slabs_gradient_parameters (slabs_grad_data_t *grad_data, slabs_inverse_problem_params_t *inverse_params)

{
  slabs_gradient_type_t *gradient_param;

  if (gradient_param == SL_GRADIENT_WEAKFACTOR)
    YMIR_GLOBAL_PRODUCTION ("Computing gradient of the weakfactor.\n");
  if (gradient_param == SL_GRADIENT_STRAIN_RATE_EXPONENT)
    YMIR_GLOBAL_PRODUCTION ("Computing gradient of the weakfactor.\n");
  if (gradient_param == SL_GRADIENT_UPPER_MANTLE_PREFACTOR)
    YMIR_GLOBAL_PRODUCTION ("Computing gradient of the weakfactor.\n");
  if (gradient_param == SL_GRADIENT_TRANSITION_ZONE_PREFACTOR)
    YMIR_GLOBAL_PRODUCTION ("Computing gradient of the weakfactor.\n");
  if (gradient_param == SL_GRADIENT_YIELD_STRESS)
    YMIR_GLOBAL_PRODUCTION ("Computing gradient of the weakfactor.\n");
  if (gradient_param == SL_GRADIENT_ACTIVATION_ENERGY)
    YMIR_GLOBAL_PRODUCTION ("Computing gradient of the weakfactor.\n");

 }
#endif

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



#if 0
void
slabs_compute_transition_zone_prefactor_prior (slabs_inverse_problem_params_t *inverse_params,
					    const char *vtk_filepath)

{
  const char *this_fn_name = "Computing the gradient of the weakzone prefactor.\n";
#if 1
  ymir_mesh_t *mesh = inverse_params->mesh;
  ymir_dvec_t *transition_zone_marker = inverse_params->transition_zone_marker;
  ymir_dvec_t *transition_zone_mean = ymir_vec_template (transition_zone_marker);
  ymir_dvec_t *transition_zone_prior_misfit = inverse_params->transition_zone_prefactor_prior_misfit;
  ymir_dvec_t *transition_zone_distribution = ymir_vec_template (transition_zone_marker);
  double       transition_zone_prefactor = inverse_params->tprior;

  char                path[BUFSIZ];

  
  snprintf (path, BUFSIZ, "%s_grad_strain_rate_exponent", vtk_filepath);
  ymir_vec_scale (inverse_params->transition_zone_prefactor, transition_zone_distribution);
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

#endif




void 
slabs_solve_optimization (slabs_inverse_problem_params_t *inverse_params)
{

  const char *this_fn_name = "entering solving optimization problem.\n";
  double                cost = inverse_params->misfit_plate_vel;
  ymir_mesh_t           *mesh = inverse_params->mesh;
  ymir_pressure_elem_t  *press_elem = inverse_params->press_elem;
  ymir_vec_t            *uobs = inverse_params->velocity_data_vec;
  ymir_vec_t            *adjoint_vq = inverse_params->adjoint_vq;
  ymir_vec_t            *u_forward = inverse_params->u;
  ymir_vec_t            *eu = inverse_params->eu;
  ymir_vec_t            *ev = inverse_params->ev;
  ymir_vec_t            *euev = inverse_params->euev;

  PetscScalar value[3];
  double up[3];
  value[0]=0;
  value[1] = 1;
  value[2] = 2;
  up[0]= value[0];
  up[1] = value[1];
  up[2] = value[2];

#if 0
    double                grad_proj[1], hessian[1], update[1];
    double                proj_grad_prefactor, proj_grad_strain_exp;
    double                proj_grad_yield_stress;
    double                proj_weak_topog;
    double                state_rtol;
    double                adjoint_rtol;
    double                inc_rtol;
    double                cg_rtol = 0.5;
    double                cg_rtol_min;
    double                cg_rtol_max;
    double                lsalpha, lsmin, lsmax;
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
    /* grad_proj[1] = 0; */
    /* grad_proj[2] = 0; */

    hessian[0] = 0;
    /* hessian[1] = 0; */
    /* hessian[2] = 0; */
    /* hessian[3] = 0; */
    /* hessian[4] = 0; */
    /* hessian[5] = 0; */
    /* hessian[6] = 0; */
    /* hessian[7] = 0; */
    /* hessian[8] = 0; */

    
    update[0] = 0;
    /* update[1] = 0; */
    /* update[2] = 0; */


    /* Compute adjoint solution where the RHS is the misfit in surface
       velocities.
    */
    
    

    YMIR_GLOBAL_PRODUCTION ("Entering Optimization Routine.\n");
    ymir_vec_set_zero (adjoint_vq);
    YMIR_GLOBAL_PRODUCTION ("Set adjoint rhs to zero.\n");
    
    /* Compute cost functional */
    YMIR_GLOBAL_PRODUCTIONF ("Optimization cost functional %E\n", cost);


    

    
    
    /* Compute the gradient: need to make option for different parameter inversion
     */
 
    /* slabs_compute_gradient_weakfactor (grad_weak_factor, state, adjoint_vq, */
    /* 				       physics_options, lin_stokes, nl_stokes, */
    /* 				       mesh, press_elem, NULL); */
    
 

 
    
    ymir_vec_scale (-1.0, grad_strain_exp);

    /* Need to project gradient */
    /* proj_grad_prefactor = ymir_vec_innerprod (grad_weak_factor, weak_zone_stencil); */
    /* proj_grad_prefactor += proj_weak_topog; */
    /* proj_grad_strain_exp = ymir_vec_innerprod (grad_strain_exp, upper_mantle_stencil); */
    /* proj_grad_yield_stress = ymir_vec_innerprod (grad_yield_stress, ones); */


    /* grad_proj[0] = -proj_grad_prefactor; */
    grad_proj[0] = -proj_grad_strain_exp;
    /* grad_proj[2] = -proj_grad_yield_stress; */

    /* YMIR_GLOBAL_PRODUCTIONF ("gradient of prefactor: %g.\n", grad_proj[0]); */
    YMIR_GLOBAL_PRODUCTIONF ("gradient of strain exp: %g.\n", grad_proj[0]);
    /* YMIR_GLOBAL_PRODUCTIONF ("gradient of yield stress: %g.\n", grad_proj[2]); */
	

    

    
    /* Note this is the weak factor stencil: computed only once */
    slabs_physics_compute_weak_factor_distrib (weak_factor, physics_options);
    slabs_physics_compute_UM_distrib (upper_mantle, physics_options);
  
    YMIR_GLOBAL_VERBOSEF ("GRadient norm: %E\n", gradnorm);
    
    
    for (i = 0; i < 10; i++) {
      
      if (!i) {
	orignorm = gradnorm;
      }
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
      slabs_hessian_topog_construct_explicit (mesh, &hessian, inverse_params);
      update[0] = grad_proj[0] / hessian[0];
      /* slabs_solve_hessian3d_direct (&update, &hessian, &grad_proj); */
#endif      
      /* If outer iteration > 1 then destroy previous adjoint/Newton operators and
	 preconditioners */
      if (i > 100){	      
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
      /* old_prefactor = physics_options->toy_weakzone_plate1; */
      /* old_yield_stress = physics_options->viscosity_stress_yield; */
      lsalpha = 1.0;
      
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
      /* proj_gradient_prefactor =  -grad_proj[0]; */
      proj_gradient_strain_exp = -grad_proj[0];
      /* proj_gradient_yield_stress = -grad_proj[2]; */


      YMIR_GLOBAL_PRODUCTIONF ("Update for the prefactor is: %g.\n", update[0]);
      /* YMIR_GLOBAL_PRODUCTIONF ("Update for the strain rate exponent is: %g.\n", update[1]); */
      /* YMIR_GLOBAL_PRODUCTIONF ("Update for the yield stress is: %g.\n", update[2]); */


      do {
	double prefactor;
	double strain_exp;
	double yield_stress;
	double rescale = 1.0;
        double    prefactor_scale = 1.0;
	double    strain_exp_scale = 1.0;

	if (i < 1){
	  do {
	    update[0] = update[0] / 2.;
	  } while ( abs(update[0]) > 100.);
	}

	/* prefactor = log (old_prefactor); */
	/* prefactor += lsalpha * prefactor_scale   * update[0]; */
	/* need to transform the 'log prefactor' to exp(prefactor)' */
	/* prefactor = exp (prefactor); */
	/* YMIR_GLOBAL_PRODUCTIONF ("old prefactor is: %g, New prefactor is: %g.\n", old_prefactor, prefactor); */

	strain_exp = log (old_strain_exp);
	strain_exp += lsalpha * strain_exp_scale * update[0];
	/* need to transform the 'log prefactor' to exp(prefactor)' */
	strain_exp = exp (strain_exp);
	YMIR_GLOBAL_PRODUCTIONF ("old strain exp is: %g, New strain rate exponent is: %g.\n",
				 old_strain_exp, strain_exp);


	/* yield_stress = log (old_yield_stress); */
	/* yield_stress += lsalpha * update[2]; */
	/* /\* need to transform the 'log prefactor' to exp(prefactor)' *\/ */
	/* yield_stress = exp (yield_stress); */
	/* YMIR_GLOBAL_PRODUCTIONF ("old yield stress is: %g, New yield stress is: %g.\n",  */
	/* 			 old_yield_stress, yield_stress); */


	/* recompute weak zone. First we set the velocity-pressure vec to zero and change
	   the vale of the prefactor. 
	*/
       	double prefactor_new = prefactor;
	double strain_exp_new = strain_exp;
	double yield_stress_new = yield_stress;
	ymir_vec_set_zero (state->vel_press_vec);
	/* slabs_update_prefactor_simple (prefactor_new, physics_options); */
 	slabs_update_strain_exp (strain_exp_new, physics_options);
	/* slabs_update_yield_stress (yield_stress_new, physics_options); */
	/* slabs_physics_compute_weakzone (state->weak_vec, physics_options); */
  
	/* Create new stokes problem if we need to backtrack */
     
	nl_stokes_resolve = slabs_nonlinear_stokes_problem_new (state, mesh, press_elem,
								physics_options);
	
	slabs_solve_stokes (lin_stokes, &nl_stokes_resolve, p8est, &mesh, &press_elem,
			    state, physics_options, discr_options,
			    solver_options, NULL, NULL, NULL);
	slabs_nonlinear_stokes_op_new (nl_stokes_resolve, state, physics_options, 0,
				       solver_options->nl_solver_type,
				       solver_options->nl_solver_primaldual_type,
				       solver_options->nl_solver_primaldual_scal_type, NULL);
  
	slabs_nonlinear_stokes_pc_new (nl_stokes_resolve, state,
				       solver_options->schur_diag_type,
				       solver_options->scaling_type, 0, NULL);

	slabs_residual_topog (topog, mesh, nl_stokes_resolve, state,
			      physics_options, NULL);
	slabs_nonlinear_stokes_op_pc_destroy (nl_stokes_resolve);

	inverse_params->topog = topog;
	slabs_compute_topog_misfit (topog_misfit, topog_obs, topog, state);
	inverse_params->topog_misfit = topog_misfit;
	/* Now compute new misfit in surface velocities */      
	new_cost = slabs_compute_dynamic_topography_misfit (topog_obs, topog, state);

	YMIR_GLOBAL_VERBOSEF ("line search: alpha %f old cost %E new cost %E\n",
			      lsalpha, cost, new_cost);
      

	/* Use globalization via Armijo Line search: If (J(m_k+1)<J(m_k)+c*alpha*d*g^T), we 
	   accept the new solution (as well as the operator used to compute next adjoint
	   solution). */

	if (new_cost < (cost - lsalpha * armijo_c1 * ( proj_gradient_strain_exp * update[0]))){

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
	  /* inverse_params->yield_stress = yield_stress; */
	  inverse_params->stokes_pc = nl_stokes_resolve->stokes_pc;
	  inverse_params->yielding_marker = nl_stokes_resolve->yielding_marker;
	  inverse_params->bounds_marker = nl_stokes_resolve->bounds_marker;
	  inverse_params->topog = topog;
	  inverse_params->topog_misfit = topog_misfit;

	  YMIR_GLOBAL_PRODUCTIONF (" The new prefactor is: %g.\n", inverse_params->prefactor);
	  YMIR_GLOBAL_PRODUCTIONF (" The new strain rate exponent is: %g.\n", inverse_params->strain_exp);
	  /* YMIR_GLOBAL_PRODUCTIONF (" The new yield stress is: %g.\n", inverse_params->yield_stress); */

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
      slabs_solve_adjoint_topography (adjoint_vq, state, 
				      topog_obs, topog,  lin_stokes,
				      nl_stokes_resolve, press_elem,
				      physics_options,
				      discr_options,
				      solver_options, NULL,
				      NULL);

      /* Now compute the gradient to be used as the RHS for Hessian system */
      ymir_vec_set_zero (grad);
      /* slabs_compute_gradient_weakfactor (grad_weak_factor, state, adjoint_vq, */
      /* 					 physics_options, lin_stokes, */
      /* 					 nl_stokes_resolve, */
      /* 					 mesh, press_elem, NULL); */
 


      /* proj_weak_topog = slabs_compute_gradient_weakfactor_topog (grad_weak_surface,  */
      /* 								 proj_weak_topog, */
      /* 								 state, */
      /* 								 topog_obs, */
      /* 								 topog, */
      /* 								 physics_options, */
      /* 								 nl_stokes_resolve, */
      /* 								 mesh, */
      /* 								 press_elem, */
      /* 								 NULL); */

      proj_weak_topog = slabs_compute_gradient_strain_exp_topog (grad_weak_surface, 
								 proj_weak_topog,
								 state,
								 topog_obs,
								 topog,
								 physics_options,
								 inverse_params,
								 nl_stokes_resolve,
								 mesh,
								 press_elem,
								 NULL);



      slabs_compute_gradient_strain_rate_exponent (grad_strain_exp, state, adjoint_vq,
      						   physics_options, lin_stokes,
      						   &nl_stokes_resolve,
      						   mesh, press_elem, NULL);
      ymir_vec_scale (-1.0, grad_strain_exp);
 

      /* slabs_compute_gradient_yield_stress (grad_yield_stress, state, adjoint_vq, */
      /* 					   physics_options, lin_stokes, */
      /* 					   &nl_stokes_resolve, */
      /* 					   mesh, press_elem, NULL); */
 


      /* Need to project gradient */
    /* proj_grad_prefactor = ymir_vec_innerprod (grad_weak_factor, weak_zone_stencil); */
      proj_grad_strain_exp = ymir_vec_innerprod (grad_strain_exp, upper_mantle_stencil);
      proj_grad_strain_exp -= proj_weak_topog;
    /* proj_grad_strain_exp = ymir_vec_innerprod (grad_strain_exp, upper_mantle_stencil); */
    /* proj_grad_yield_stress = ymir_vec_innerprod (grad_yield_stress, ones); */

    /* proj_grad_prefactor += proj_weak_topog; */
    /*  grad_proj[0] = -proj_grad_prefactor; */
    grad_proj[0] = -proj_grad_strain_exp;
    /* grad_proj[2] = -proj_grad_yield_stress; */


    YMIR_GLOBAL_PRODUCTIONF ("gradient of prefactor: %g.\n", grad_proj[0]);
    /* YMIR_GLOBAL_PRODUCTIONF ("gradient of strain exp: %g.\n", grad_proj[1]); */
    /* YMIR_GLOBAL_PRODUCTIONF ("gradient of strain yield stress: %g.\n", grad_proj[2]); */
	

    
}

    /* Final destruction of adjoint/Newton operators and nl-stokes problem */
    slabs_nonlinear_stokes_op_pc_destroy (nl_stokes_resolve);
    slabs_nonlinear_stokes_problem_destroy (nl_stokes_resolve);

#endif  
    YMIR_GLOBAL_INFOF ("Finished %s", this_fn_name);


}


