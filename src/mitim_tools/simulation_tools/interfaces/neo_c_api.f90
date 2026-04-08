!---------------------------------------------------------------------------
! neo_c_api.f90
!
! PURPOSE:
!   Thin Fortran wrapper exposing NEO subroutines and output variables
!   through C-compatible interfaces (iso_c_binding).  This file lives in
!   MITIM-fusion and is compiled together with the NEO sources to produce
!   libneo_serial.so, loaded by neo_inprocess.py via ctypes.
!
! NOTES:
!   * NEO is built with -fdefault-real-8, so Fortran REAL == 8 bytes ==
!     c_double, matching all the neo_*_in / neo_*_out module variables.
!   * Per-species output arrays are dimension(11) (n_species_max).
!   * Geometry output array is dimension(5).
!   * neo_init_serial(path) sets path/i_proc/n_proc and stubs the MPI
!     communicator (NEO_COMM_WORLD = -1) so neo_do() runs serially.
!   * Two entry points are exposed:
!       - c_neo_run_file()  : read input.neo.gen from disk, then run
!       - c_neo_run_iface() : run with whatever was set in neo_*_in
!---------------------------------------------------------------------------

module neo_c_api

  use iso_c_binding
  use neo_interface
  use neo_globals, only: path, silent_flag, i_proc, n_proc, NEO_COMM_WORLD

  implicit none

contains

  ! -------------------------------------------------------------------------
  ! c_neo_set_path
  !   Initialise NEO for serial use and set the working directory path.
  !   `path_cstr` is a null-terminated C string ending with '/'.
  ! -------------------------------------------------------------------------
  subroutine c_neo_set_path(path_cstr) bind(C, name="c_neo_set_path")
    character(kind=c_char), dimension(*), intent(in) :: path_cstr
    integer :: i

    path = ' '
    do i = 1, len(path)
      if (path_cstr(i) == c_null_char) exit
      path(i:i) = path_cstr(i)
    end do

    ! Mimic neo_init_serial: serial execution, fake communicator.
    i_proc         = 0
    n_proc         = 1
    NEO_COMM_WORLD = -1

    ! Always quiet.
    neo_silent_flag_in = 1

  end subroutine c_neo_set_path

  ! -------------------------------------------------------------------------
  ! c_neo_read_input
  !   Read parameters from input.neo.gen located at the path set above
  !   and copy them into the neo_interface module variables, so the next
  !   call to c_neo_run_iface() picks them up.
  ! -------------------------------------------------------------------------
  subroutine c_neo_read_input() bind(C, name="c_neo_read_input")
    call neo_read_input()
    call map_global2interface()
  end subroutine c_neo_read_input

  ! -------------------------------------------------------------------------
  ! c_neo_run
  !   Run NEO using whatever values are currently in the neo_*_in module
  !   variables.  This is the high-level subroutine entry point: neo_run()
  !   internally calls map_interface2global() and neo_do().
  ! -------------------------------------------------------------------------
  subroutine c_neo_run() bind(C, name="c_neo_run")
    call neo_run()
  end subroutine c_neo_run

  ! -------------------------------------------------------------------------
  ! c_neo_get_outputs
  !   Copy output module variables into C-accessible arguments.  All
  !   per-species arrays have length 11 (= n_species_max).
  ! -------------------------------------------------------------------------
  subroutine c_neo_get_outputs(                                  &
       ns_out,                                                   &
       pflux_thHH, eflux_thHHi, eflux_thHHe, eflux_thCHi,        &
       jpar_thS, jpar_thK, jpar_thN, jtor_thS,                   &
       jpar_thSmod, jtor_thSmod,                                 &
       pflux_thHS, eflux_thHS,                                   &
       pflux_dke, efluxtot_dke, efluxncv_dke, mflux_dke,         &
       vpol_dke, vtor_dke, jpar_dke, jtor_dke,                   &
       pflux_gv, efluxtot_gv, efluxncv_gv, mflux_gv,             &
       nclassvis, pflux_nclass, efluxtot_nclass,                 &
       vpol_nclass, vtor_nclass, jpar_nclass,                    &
       geoparams,                                                &
       error_status                                              &
       ) bind(C, name="c_neo_get_outputs")

    integer(c_int), intent(out) :: ns_out
    real(c_double), intent(out) :: pflux_thHH, eflux_thHHi, eflux_thHHe, eflux_thCHi
    real(c_double), intent(out) :: jpar_thS, jpar_thK, jpar_thN, jtor_thS
    real(c_double), intent(out) :: jpar_thSmod, jtor_thSmod
    real(c_double), intent(out) :: pflux_thHS(11), eflux_thHS(11)
    real(c_double), intent(out) :: pflux_dke(11), efluxtot_dke(11), efluxncv_dke(11)
    real(c_double), intent(out) :: mflux_dke(11), vpol_dke(11), vtor_dke(11)
    real(c_double), intent(out) :: jpar_dke, jtor_dke
    real(c_double), intent(out) :: pflux_gv(11), efluxtot_gv(11), efluxncv_gv(11), mflux_gv(11)
    real(c_double), intent(out) :: nclassvis(11), pflux_nclass(11), efluxtot_nclass(11)
    real(c_double), intent(out) :: vpol_nclass(11), vtor_nclass(11)
    real(c_double), intent(out) :: jpar_nclass
    real(c_double), intent(out) :: geoparams(5)
    integer(c_int), intent(out) :: error_status

    ns_out          = neo_n_species_in

    pflux_thHH      = neo_pflux_thHH_out
    eflux_thHHi     = neo_eflux_thHHi_out
    eflux_thHHe     = neo_eflux_thHHe_out
    eflux_thCHi     = neo_eflux_thCHi_out
    jpar_thS        = neo_jpar_thS_out
    jpar_thK        = neo_jpar_thK_out
    jpar_thN        = neo_jpar_thN_out
    jtor_thS        = neo_jtor_thS_out
    jpar_thSmod     = neo_jpar_thSmod_out
    jtor_thSmod     = neo_jtor_thSmod_out

    pflux_thHS      = neo_pflux_thHS_out
    eflux_thHS      = neo_eflux_thHS_out

    pflux_dke       = neo_pflux_dke_out
    efluxtot_dke    = neo_efluxtot_dke_out
    efluxncv_dke    = neo_efluxncv_dke_out
    mflux_dke       = neo_mflux_dke_out
    vpol_dke        = neo_vpol_dke_out
    vtor_dke        = neo_vtor_dke_out
    jpar_dke        = neo_jpar_dke_out
    jtor_dke        = neo_jtor_dke_out

    pflux_gv        = neo_pflux_gv_out
    efluxtot_gv     = neo_efluxtot_gv_out
    efluxncv_gv     = neo_efluxncv_gv_out
    mflux_gv        = neo_mflux_gv_out

    nclassvis       = neo_nclassvis_out
    pflux_nclass    = neo_pflux_nclass_out
    efluxtot_nclass = neo_efluxtot_nclass_out
    vpol_nclass     = neo_vpol_nclass_out
    vtor_nclass     = neo_vtor_nclass_out
    jpar_nclass     = neo_jpar_nclass_out

    geoparams       = neo_geoparams_out

    error_status    = neo_error_status_out

  end subroutine c_neo_get_outputs

end module neo_c_api
