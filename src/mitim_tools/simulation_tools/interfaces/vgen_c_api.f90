!---------------------------------------------------------------------------
! vgen_c_api.f90
!
! PURPOSE:
!   Thin Fortran C-binding wrapper that runs the gacode "vgen" velocity-
!   generation workflow in-process.  Built into libvgen_serial.so and
!   loaded from Python via ctypes (vgen_inprocess.py).
!
!   The standard `profiles_gen -vgen` driver is the `vgen` Fortran program
!   in gacode/vgen/src/vgen.f90, which:
!     1. Calls neo_init_serial → neo_read_input → expro_read('input.gacode')
!     2. Distributes radial surfaces across MPI ranks
!     3. Calls vgen_compute_neo on each surface (which sets neo_*_in,
!        runs neo_run, reads neo_*_out)
!     4. Computes Er, w0, w0p from the per-surface results
!     5. Writes a new input.gacode via expro_write
!
!   This wrapper does (1)–(5) **without** any MPI: it loops over surfaces
!   sequentially in a single thread, calls expro_read with comm=0 to disable
!   MPI inside expro, and skips vgen_reduce entirely (no need with one rank).
!
!   For the PORTALS use case (er_method = 2 = NEO weak rotation limit, with
!   zero toroidal rotation) the only output that matters is EXPRO_w0,
!   which is written into vgen/input.gacode by expro_write.
!---------------------------------------------------------------------------

module vgen_c_api

  use iso_c_binding
  use neo_interface
  use neo_globals,   only: path, silent_flag, i_proc, n_proc, NEO_COMM_WORLD
  use vgen_globals,  only: vgen_path => path, &
                           vgen_i_proc => i_proc, vgen_n_proc => n_proc, &
                           er_method, vel_method, erspecies_indx, epar_flag, &
                           nth_min, nth_max, nn_flag, &
                           dens_norm, temp_norm, mass_norm, vth_norm, jbs_norm, e_norm, &
                           vtor_measured, &
                           pflux_sum, &
                           jbs_neo, jsigma_neo, jtor_neo, &
                           jbs_sauter, jsigma_sauter, jtor_sauter, &
                           jbs_sauter_mod, jsigma_sauter_mod, jtor_sauter_mod, &
                           n_ions, tag, fmt, &
                           temp_norm_fac, charge_norm_fac
  use expro

  implicit none

contains

  ! -------------------------------------------------------------------------
  ! c_vgen_set_path
  !   Set the working directory for vgen.  expro_read / expro_write open
  !   files relative to the current working directory, so the Python wrapper
  !   chdir's into `path` before calling c_vgen_run.  This routine just
  !   stores the path string into both neo_globals::path and
  !   vgen_globals::path for any code that consults them.
  ! -------------------------------------------------------------------------
  subroutine c_vgen_set_path(path_cstr) bind(C, name="c_vgen_set_path")
    character(kind=c_char), dimension(*), intent(in) :: path_cstr
    integer :: i

    path      = ' '
    vgen_path = ' '
    do i = 1, len(path)
      if (path_cstr(i) == c_null_char) exit
      path(i:i)      = path_cstr(i)
      vgen_path(i:i) = path_cstr(i)
    end do

  end subroutine c_vgen_set_path

  ! -------------------------------------------------------------------------
  ! c_vgen_run
  !   Run the full vgen workflow on input.gacode in the current directory.
  !
  !   Arguments:
  !     er_method_in     : 1 = force balance (needs vpol/vtor), 2 = NEO weak
  !                        rotation limit (recommended for zero Vtor),
  !                        4 = use given omega0
  !     vel_method_in    : 1 = NEO weak rotation, 2 = NEO strong rotation
  !     erspecies_indx_in: Index (1-based) of the ion species to match
  !     nth_min_in,
  !     nth_max_in       : Min / max poloidal theta resolution
  !     n_species_in     : Number of NEO species (set in neo_n_species_in)
  !
  !   On return, the new input.gacode (with populated EXPRO_w0) has been
  !   written to <cwd>/vgen/input.gacode.  The Python wrapper is responsible
  !   for ensuring the cwd already contains an input.gacode file and an
  !   empty vgen/ subdirectory.
  ! -------------------------------------------------------------------------
  subroutine c_vgen_run(er_method_in, vel_method_in, erspecies_indx_in, &
                        nth_min_in, nth_max_in, n_species_in)            &
                        bind(C, name="c_vgen_run")

    integer(c_int), intent(in), value :: er_method_in
    integer(c_int), intent(in), value :: vel_method_in
    integer(c_int), intent(in), value :: erspecies_indx_in
    integer(c_int), intent(in), value :: nth_min_in
    integer(c_int), intent(in), value :: nth_max_in
    integer(c_int), intent(in), value :: n_species_in

    integer :: i, j, ix, rotation_model, simntheta, iteration_flag
    real    :: vtor_diff, er0, omega, omega_deriv, ya, yb, grad_p
    real, dimension(:), allocatable :: er_exp

    ! ---- Stash vgen control inputs into vgen_globals ----
    er_method      = er_method_in
    vel_method     = vel_method_in
    erspecies_indx = erspecies_indx_in
    nth_min        = nth_min_in
    nth_max        = nth_max_in
    epar_flag      = 0       ! conductivity calculation off
    nn_flag        = 0       ! NEO neural-net path off

    ! ---- Validate supported methods up-front, before any allocation ----
    ! Only the PORTALS use case is implemented: er_method=2 (Er from the NEO
    ! weak-rotation limit) and vel_method=1 (weak-rotation NEO flows).
    ! vgen.f90's other branches (force balance, strong-rotation second pass)
    ! are not ported; failing loudly here beats silently returning
    ! weak-rotation results for a strong-rotation request.  Returning before
    ! the allocations also keeps a failed call re-entrant (no allocated
    ! arrays left behind to crash the next call in the same process).
    if (er_method /= 2) then
       print '(a,i0)', 'ERROR: (VGEN c_api) only er_method=2 is supported, got ', er_method
       return
    endif
    if (vel_method /= 1) then
       print '(a,i0)', 'ERROR: (VGEN c_api) only vel_method=1 (weak rotation) is supported, got ', vel_method
       return
    endif

    ! ---- Serial init: pretend we're rank 0 of a 1-rank communicator ----
    i_proc          = 0
    n_proc          = 1
    NEO_COMM_WORLD  = -1
    vgen_i_proc     = 0
    vgen_n_proc     = 1

    ! Initialise the per-rank file-tag array (vgen_compute_neo writes
    ! out.vgen.neontheta00 etc.; the main vgen.f90 fills `tag` in its main
    ! loop, but we are bypassing that, so do it here.)
    do ix = 1, 100
       write(tag(ix), fmt) ix - 1
    end do

    ! ---- Mirror neo_init_serial: set neo_globals state for serial use ----
    silent_flag = 1

    ! ---- Set NEO interface variables directly to the same values that
    !      gacode/vgen/templates/input.neo.default specifies, plus the
    !      ones profiles_gen -vgen appends afterwards (SUBROUTINE_FLAG=1,
    !      EQUILIBRIUM_MODEL=2, N_SPECIES=...).  We bypass neo_read_input
    !      entirely so the wrapper has zero file dependencies on input.neo*.
    neo_n_energy_in         = 6
    neo_n_xi_in             = 17
    neo_n_theta_in          = 17
    neo_sim_model_in        = 2
    neo_ae_flag_in          = 0
    neo_equilibrium_model_in = 2
    neo_subroutine_flag     = 1
    neo_n_species_in        = n_species_in
    neo_n_radial_in         = 1
    neo_profile_model_in    = 1
    neo_silent_flag_in      = 1

    EXPRO_ctrl_quasineutral_flag = 1
    EXPRO_ctrl_n_ion             = neo_n_species_in + neo_ae_flag_in - 1
    n_ions                       = EXPRO_ctrl_n_ion

    ! comm=0 → expro_read sets hasmpi=.false. and skips all MPI broadcasts
    call expro_read('input.gacode', 0)

    if (EXPRO_error == 1) then
       print '(a)', 'ERROR: (VGEN) Negative ion density'
       return
    endif

    ! Set ion masses and charges from EXPRO
    do j = 1, EXPRO_ctrl_n_ion
       neo_z_in(j)    = expro_z(j)
       neo_mass_in(j) = expro_mass(j) / 2.0
    end do
    if (neo_ae_flag_in == 0) then
       neo_z_in(neo_n_species_in)    = expro_ze
       neo_mass_in(neo_n_species_in) = expro_masse / 2.0
    endif

    ! Set sign of btccw and ipccw from sign of B and q from EXPRO
    neo_btccw_in = -EXPRO_signb
    neo_ipccw_in = -EXPRO_signb * EXPRO_signq

    ! Working storage that vgen_init normally allocates
    allocate(vtor_measured  (EXPRO_n_exp))
    allocate(pflux_sum      (EXPRO_n_exp))
    allocate(jbs_neo        (EXPRO_n_exp))
    allocate(jsigma_neo     (EXPRO_n_exp))
    allocate(jtor_neo       (EXPRO_n_exp))
    allocate(jbs_sauter     (EXPRO_n_exp))
    allocate(jsigma_sauter  (EXPRO_n_exp))
    allocate(jtor_sauter    (EXPRO_n_exp))
    allocate(jbs_sauter_mod (EXPRO_n_exp))
    allocate(jsigma_sauter_mod(EXPRO_n_exp))
    allocate(jtor_sauter_mod  (EXPRO_n_exp))

    pflux_sum         = 0.0
    jbs_neo           = 0.0
    jsigma_neo        = 0.0
    jtor_neo          = 0.0
    jbs_sauter        = 0.0
    jsigma_sauter     = 0.0
    jtor_sauter       = 0.0
    jbs_sauter_mod    = 0.0
    jsigma_sauter_mod = 0.0
    jtor_sauter_mod   = 0.0

    do j = 1, EXPRO_n_ion
       if (j == erspecies_indx) then
          vtor_measured(:) = EXPRO_vtor(j, :)
       endif
       EXPRO_vpol(j, :) = 0.0
       EXPRO_vtor(j, :) = 0.0
    end do

    ! ---- Allocate Er working array ----
    allocate(er_exp(EXPRO_n_exp))
    if (er_method /= 4) then
       er_exp(:)    = 0.0
       EXPRO_w0(:)  = 0.0
       EXPRO_w0p(:) = 0.0
    endif

    ! ====================================================================
    ! Sequential per-surface loop (replaces the MPI-distributed loop in
    ! vgen.f90).  Only er_method=2 / vel_method=1 are supported here
    ! (validated up-front, before any allocation) — that is what PORTALS
    ! uses for the neoclassical ExB shear (zero toroidal rotation, NEO
    ! weak rotation limit).  Other methods would require additional logic;
    ! add it here if needed.
    ! ====================================================================
    do i = 2, EXPRO_n_exp - 1
       rotation_model = 1            ! weak rotation
       er0            = 0.0
       omega          = 0.0
       omega_deriv    = 0.0
       iteration_flag = 1

       call vgen_compute_neo(i, vtor_diff, rotation_model, er0, &
                             omega, omega_deriv, simntheta, iteration_flag)

       ! Same Er-from-vtor_diff formula as vgen.f90 case(2)
       er_exp(i) = (vtor_diff / (vth_norm * EXPRO_rmin(EXPRO_n_exp)))     &
            / (neo_rmaj_over_a_in + neo_rmin_over_a_in)                   &
            * neo_rmin_over_a_in / (abs(neo_q_in) * EXPRO_signq)          &
            / (abs(neo_rho_star_in) * EXPRO_signb)                        &
            * EXPRO_grad_r0(i)                                            &
            * (temp_norm * temp_norm_fac / charge_norm_fac                &
               / EXPRO_rmin(EXPRO_n_exp)) / 1000

       ! Add the Er-driven contribution to vtor for every ion
       do j = 1, n_ions
          EXPRO_vtor(j, i) = EXPRO_vtor(j, i) + vtor_diff
       end do
    end do

    ! ====================================================================
    ! Compute EXPRO_w0 / EXPRO_w0p and extrapolate to the boundary points.
    ! Mirrors the post-loop block of vgen.f90.
    ! ====================================================================
    do i = 2, EXPRO_n_exp - 1
       EXPRO_w0(i) = 2.9979e10 * EXPRO_q(i) * (er_exp(i) / 30.0) /         &
            ((1e4 * EXPRO_bunit(i)) * (1e2 * EXPRO_rmin(i)) * EXPRO_grad_r0(i))
    end do

    call bound_extrap(ya, yb, er_exp,    EXPRO_rmin, EXPRO_n_exp)
    er_exp(1)           = ya
    er_exp(EXPRO_n_exp) = yb

    call bound_extrap(ya, yb, EXPRO_w0,  EXPRO_rmin, EXPRO_n_exp)
    EXPRO_w0(1)           = ya
    EXPRO_w0(EXPRO_n_exp) = yb

    call bound_deriv(EXPRO_w0p(2:EXPRO_n_exp-1), EXPRO_w0(2:EXPRO_n_exp-1), &
                     EXPRO_rmin, EXPRO_n_exp - 2)
    call bound_extrap(ya, yb, EXPRO_w0p, EXPRO_rmin, EXPRO_n_exp)
    EXPRO_w0p(1)           = ya
    EXPRO_w0p(EXPRO_n_exp) = yb

    do j = 1, n_ions
       call bound_extrap(ya, yb, EXPRO_vpol(j, :), EXPRO_rmin, EXPRO_n_exp)
       EXPRO_vpol(j, 1)           = ya
       EXPRO_vpol(j, EXPRO_n_exp) = yb
       call bound_extrap(ya, yb, EXPRO_vtor(j, :), EXPRO_rmin, EXPRO_n_exp)
       EXPRO_vtor(j, 1)           = ya
       EXPRO_vtor(j, EXPRO_n_exp) = yb
    end do

    ! ====================================================================
    ! Boundary-extrapolate and assign the bootstrap current, toroidal
    ! current and parallel conductivity computed by vgen_compute_neo,
    ! mirroring vgen.f90 (neo_sim_model_in = 2 here, so the NEO arrays are
    ! the live branch).  Without this the output file silently keeps the
    ! INPUT file's jbs/jbstor/sigmapar columns.
    ! ====================================================================
    call bound_extrap(ya, yb, jbs_neo,    EXPRO_rmin, EXPRO_n_exp)
    jbs_neo(1)              = ya
    jbs_neo(EXPRO_n_exp)    = yb
    call bound_extrap(ya, yb, jtor_neo,   EXPRO_rmin, EXPRO_n_exp)
    jtor_neo(1)             = ya
    jtor_neo(EXPRO_n_exp)   = yb
    call bound_extrap(ya, yb, jsigma_neo, EXPRO_rmin, EXPRO_n_exp)
    jsigma_neo(1)           = ya
    jsigma_neo(EXPRO_n_exp) = yb

    EXPRO_jbs(:)      = jbs_neo(:)
    EXPRO_jbstor(:)   = jtor_neo(:)
    EXPRO_sigmapar(:) = jsigma_neo(:)

    ! ====================================================================
    ! Write the new input.gacode (vgen/input.gacode), the same way the
    ! standard vgen driver does.  expro_write opens this path relative to
    ! the cwd that c_vgen_set_path established.
    ! ====================================================================
    expro_head_vgen = '#      vgen : in-process (er=2, weak rotation)'
    call expro_write('vgen/input.gacode')

    ! ---- Cleanup ----
    deallocate(er_exp)
    deallocate(vtor_measured)
    deallocate(pflux_sum)
    deallocate(jbs_neo)
    deallocate(jsigma_neo)
    deallocate(jtor_neo)
    deallocate(jbs_sauter)
    deallocate(jsigma_sauter)
    deallocate(jtor_sauter)
    deallocate(jbs_sauter_mod)
    deallocate(jsigma_sauter_mod)
    deallocate(jtor_sauter_mod)

  end subroutine c_vgen_run

end module vgen_c_api
