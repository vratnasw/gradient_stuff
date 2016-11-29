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

#ifndef SLABS_HESSIAN_H
#define SLABS_HESSIAN_H

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




/* Hessian problem structure */
typedef struct slabs_hessian_params
{
  /* define incremental vectors (second order adjoint variables and parameters) 
   */
  ymir_vec_t *up_inc;
  ymir_vec_t *vq_inc;
  ymir_vec_t *upout_inc;
  ymir_vec_t *v;
  ymir_vec_t *u_inc;
  ymir_vec_t *v_inc;
  ymir_vec_t *eu_inc;
  ymir_vec_t *ev_inc;
  ymir_vec_t *uout;
  ymir_vec_t *weakfactor_inc; 
  ymir_vec_t *strain_rate_exp_inc;
  ymir_vec_t *yield_stress_inc;
  ymir_vec_t *upper_mantle_prefactor_inc;
  ymir_vec_t *transition_zone_prefactor_inc;
  ymir_vec_t *activation_energy_inc;
  ymir_vec_t *eueu_inc;
  ymir_vec_t *euev;
  ymir_vec_t *euinc_ev;
  ymir_vec_t *weakfactor_mm_average_viscosity_inc;
  ymir_vec_t *strain_rate_exponent_mm_average_viscosity_inc;
  ymir_vec_t *yield_stress_mm_average_viscosity_inc;
  ymir_vec_t *upper_mantle_prefactor_mm_average_viscosity_inc;
  ymir_vec_t *transition_zone_prefactor_mm_average_viscosity_inc;
  ymir_vec_t *activation_energy_mm_average_viscosity_inc;
  ymir_vec_t *hessian_block_uu_average_viscosity_inc;
  ymir_vec_t *hessian_block_uu_newton_inc;
  ymir_vec_t *hessian_block_mm_UM_prefactor;
  ymir_vec_t *hessian_block_mm_strain_rate_exponent;
  ymir_vec_t *hessian_block_mm_weakfactor;
  ymir_vec_t *hessian_block_mm_yield_stress;
  ymir_vec_t *hessian_block_mm_TZ_prefactor;
  ymir_vec_t *hessian_block_mm_activation_energy;

 
  double     *hessian;

}
slabs_hessian_params_t;


void
slabs_hessian_block_uu (slabs_inverse_problem_params_t *inverse_params,
			slabs_hessian_params_t *hessian_params);


void
slabs_hessian_block_uu_newton (slabs_inverse_problem_params_t *inverse_params,
			       slabs_hessian_params_t *hessian_params);


void
slabs_hessian_block_mm_strain_exp (slabs_inverse_problem_params_t *inverse_params,
				   slabs_hessian_params_t *hessian_params);

void
slabs_hessian_block_mm_upper_mantle (slabs_inverse_problem_params_t *inverse_params,
				     slabs_hessian_params_t *hessian_params);


void
slabs_hessian_block_mm_transition_zone (slabs_inverse_problem_params_t *inverse_params,
					slabs_hessian_params_t *hessian_params);


void
slabs_hessian_block_mm_weakfactor (slabs_inverse_problem_params_t *inverse_params,
				   slabs_hessian_params_t *hessian_params);

void
slabs_hessian_block_mm_yield_stress (slabs_inverse_problem_params_t *inverse_params,
				     slabs_hessian_params_t *hessian_params);

void
slabs_hessian_block_mm_strain_exp_prior (ymir_vec_t *strain_rate_exp_inc,
					 slabs_inverse_problem_params_t *inverse_params,
					 slabs_hessian_params_t *hessian_params);

void
slabs_hessian_block_mm_weakfactor_prior (ymir_vec_t *weakfactor_inc,
					 slabs_inverse_problem_params_t *inverse_params,
					 slabs_hessian_params_t *hessian_params);
void
slabs_hessian_block_mm_upper_mantle_prior (ymir_vec_t *upper_mantle_inc,
					   slabs_inverse_problem_params_t *inverse_params,
					   slabs_hessian_params_t *hessian_params);

void
slabs_hessian_block_mm_transition_zone_prior (ymir_vec_t *transition_zone_inc,
					      slabs_inverse_problem_params_t *inverse_params,
					      slabs_hessian_params_t *hessian_params);

void
slabs_hessian_block_mm_lower_mantle_prior (ymir_vec_t *lower_mantle_inc,
					   slabs_inverse_problem_params_t *inverse_params,
					   slabs_hessian_params_t *hessian_params);

void
slabs_hessian_block_mm_yield_stress_prior (ymir_vec_t *yield_stress_inc,
					 slabs_inverse_problem_params_t *inverse_params,
					   slabs_hessian_params_t *hessian_params);

void 
slabs_hessian_block_vm_prefactor (ymir_vec_t *vq_out,
				  slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_vm_yield_stress (ymir_vec_t *vq_out,
				     slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_vm_yield_upper_mantle_prefactor (ymir_vec_t *vq_out,
						     slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_vm_yield_transition_zone_prefactor (ymir_vec_t *vq_out,
							slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_vm_yield_activation_energy (ymir_vec_t *vq_out,
						slabs_inverse_problem_params_t *inverse_params);


void 
slabs_hessian_block_vm_yield_strain_rate_exponent (ymir_vec_t *vq_out,
						   slabs_inverse_problem_params_t *inverse_params);


void 
slabs_hessian_block_mv_weakfactor (slabs_hessian_params_t *hessian_params,
				   slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_mv_strain_rate_exponent (slabs_hessian_params_t *hessian_params,
					     slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_mv_yield_stress (slabs_hessian_params_t *hessian_params,
				     slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_mv_upper_mantle_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_mv_transition_zone_prefactor (slabs_hessian_params_t *hessian_params,
						  slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_mv_activation_energy (slabs_hessian_params_t *hessian_params,
					  slabs_inverse_problem_params_t *inverse_params);

void 
slabs_hessian_block_um_weakfactor (slabs_hessian_params_t *hessian_params,
				   slabs_inverse_problem_params_t *inverse_params,
				   const char *vtk_filepath);


void 
slabs_hessian_block_um_strain_rate_exponent (slabs_hessian_params_t *hessian_params,
					     slabs_inverse_problem_params_t *inverse_params,
					     const char *vtk_filepath);

void 
slabs_hessian_block_um_yield_stress (slabs_hessian_params_t *hessian_params,
				     slabs_inverse_problem_params_t *inverse_params,
				     const char *vtk_filepath);

void 
slabs_hessian_block_um_activation_energy (slabs_hessian_params_t *hessian_params,
					  slabs_inverse_problem_params_t *inverse_params,
					  const char *vtk_filepath);
void 
slabs_hessian_block_um_upper_mantle_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params,
					       const char *vtk_filepath);


void 
slabs_hessian_block_um_transition_zone_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params,
						  const char *vtk_filepath);


void 
slabs_hessian_block_mu_weakfactor (slabs_hessian_params_t *hessian_params,
				   slabs_inverse_problem_params_t *inverse_params,
				   const char *vtk_filepath);

void 
slabs_hessian_block_mu_strain_rate_exponent (slabs_hessian_params_t *hessian_params,
					     slabs_inverse_problem_params_t *inverse_params,
					     const char *vtk_filepath);

void 
slabs_hessian_block_mu_upper_mantle_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params,
					       const char *vtk_filepath);
void 
slabs_hessian_block_mu_transition_zone_prefactor (slabs_hessian_params_t *hessian_params,
					       slabs_inverse_problem_params_t *inverse_params,
						  const char *vtk_filepath);


void 
slabs_hessian_block_uu_viscosity_average (slabs_hessian_params_t *hessian_params,
					  slabs_inverse_problem_params_t *inverse_params,
					  const char *vtk_filepath);

void 
slabs_hessian_block_mm_weakfactor_viscosity_average (slabs_hessian_params_t *hessian_params,
						     slabs_inverse_problem_params_t *inverse_params,
						     const char *vtk_filepath);


void 
slabs_hessian_block_mm_strain_rate_exponent_viscosity_average (slabs_hessian_params_t *hessian_params,
							       slabs_inverse_problem_params_t *inverse_params,
							       const char *vtk_filepath);


void 
slabs_hessian_block_mm_upper_mantle_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								 slabs_inverse_problem_params_t *inverse_params,
								 const char *vtk_filepath);

void 
slabs_hessian_block_mm_transition_zone_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								 slabs_inverse_problem_params_t *inverse_params,
								    const char *vtk_filepath);

void 
slabs_hessian_block_mu_weakfactor_viscosity_average (slabs_hessian_params_t *hessian_params,
						     slabs_inverse_problem_params_t *inverse_params,
						     const char *vtk_filepath);

void 
slabs_hessian_block_mu_strain_rate_exponent_viscosity_average (slabs_hessian_params_t *hessian_params,
							       slabs_inverse_problem_params_t *inverse_params,
							       const char *vtk_filepath);

void 
slabs_hessian_block_mu_upper_mantle_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								 slabs_inverse_problem_params_t *inverse_params,
								 const char *vtk_filepath);

void 
slabs_hessian_block_mu_transition_zone_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								    slabs_inverse_problem_params_t *inverse_params,
								    const char *vtk_filepath);


void 
slabs_hessian_block_mu_activation_energy_viscosity_average (slabs_hessian_params_t *hessian_params,
							    slabs_inverse_problem_params_t *inverse_params,
							    const char *vtk_filepath);
void 
slabs_hessian_block_mu_yield_stress_viscosity_average (slabs_hessian_params_t *hessian_params,
						       slabs_inverse_problem_params_t *inverse_params,
						       const char *vtk_filepath);

void 
slabs_hessian_block_um_weakfactor_viscosity_average (slabs_hessian_params_t *hessian_params,
						     slabs_inverse_problem_params_t *inverse_params,
						     const char *vtk_filepath);
void 
slabs_hessian_block_um_strain_rate_exponent_viscosity_average (slabs_hessian_params_t *hessian_params,
							       slabs_inverse_problem_params_t *inverse_params,
							       const char *vtk_filepath);

void 
slabs_hessian_block_um_upper_mantle_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								 slabs_inverse_problem_params_t *inverse_params,
								 const char *vtk_filepath);

void 
slabs_hessian_block_um_transition_zone_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								    slabs_inverse_problem_params_t *inverse_params,
								    const char *vtk_filepath);


void 
slabs_hessian_block_um_activation_energy_viscosity_average (slabs_hessian_params_t *hessian_params,
							    slabs_inverse_problem_params_t *inverse_params,
							    const char *vtk_filepath);

void 
slabs_hessian_block_um_yield_stress_viscosity_average (slabs_hessian_params_t *hessian_params,
						       slabs_inverse_problem_params_t *inverse_params,
						       const char *vtk_filepath);


void 
slabs_hessian_block_um_transition_zone_prefactor_viscosity_average (slabs_hessian_params_t *hessian_params,
								    slabs_inverse_problem_params_t *inverse_params,
								    const char *vtk_filepath);


void 
slabs_hessian_block_um_activation_energy_viscosity_average (slabs_hessian_params_t *hessian_params,
							    slabs_inverse_problem_params_t *inverse_params,
							    const char *vtk_filepath);


#endif /* SLABS_HESSIAN_H */
