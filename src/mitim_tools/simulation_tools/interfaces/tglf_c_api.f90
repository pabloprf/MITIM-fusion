!---------------------------------------------------------------------------
! tglf_c_api.f90
!
! PURPOSE:
!   Thin Fortran wrapper exposing TGLF subroutines and output variables
!   through C-compatible interfaces (iso_c_binding).  This file lives in
!   MITIM-fusion and is compiled together with the TGLF non-MPI sources
!   to produce libtglf_serial.so, loaded by tglf_inprocess.py via ctypes.
!
! NOTES:
!   * All real quantities come from tglf_interface which is compiled with
!     -fdefault-real-8, making Fortran REAL == 8 bytes == c_double.
!   * Arrays are sized to nsm-1 = 11 (maximum ion species count).
!   * tglf_error() calls STOP on fatal errors — only valid inputs should be
!     passed from MITIM's workflow.
!---------------------------------------------------------------------------

module tglf_c_api

  use iso_c_binding
  use tglf_interface, only: &
    tglf_path_in,           &
    tglf_quiet_flag_in,     &
    tglf_dump_flag_in,      &
    tglf_ns_in,             &
    tglf_elec_pflux_out,    &
    tglf_elec_eflux_out,    &
    tglf_elec_eflux_low_out,&
    tglf_elec_mflux_out,    &
    tglf_elec_expwd_out,    &
    tglf_ion_pflux_out,     &
    tglf_ion_eflux_out,     &
    tglf_ion_eflux_low_out, &
    tglf_ion_mflux_out,     &
    tglf_ion_expwd_out

  implicit none

contains

  ! -------------------------------------------------------------------------
  ! c_tglf_set_path
  !   Set tglf_path_in (null-terminated C string) and disable verbose output.
  ! -------------------------------------------------------------------------
  subroutine c_tglf_set_path(path_cstr) bind(C, name="c_tglf_set_path")
    character(kind=c_char), dimension(*), intent(in) :: path_cstr
    integer :: i

    tglf_path_in = ' '
    do i = 1, 256
      if (path_cstr(i) == c_null_char) exit
      tglf_path_in(i:i) = path_cstr(i)
    end do
    tglf_quiet_flag_in = .true.
    tglf_dump_flag_in  = .false.

  end subroutine c_tglf_set_path

  ! -------------------------------------------------------------------------
  ! c_tglf_read_input
  !   Read parameters from input.tglf.gen located in the path set by
  !   c_tglf_set_path.  Fortran reads: trim(tglf_path_in)//'input.tglf.gen'
  ! -------------------------------------------------------------------------
  subroutine c_tglf_read_input() bind(C, name="c_tglf_read_input")
    call tglf_read_input()
  end subroutine c_tglf_read_input

  ! -------------------------------------------------------------------------
  ! c_tglf_run
  !   Execute TGLF transport model (serial path; no file writes).
  ! -------------------------------------------------------------------------
  subroutine c_tglf_run() bind(C, name="c_tglf_run")
    call tglf_run()
  end subroutine c_tglf_run

  ! -------------------------------------------------------------------------
  ! c_tglf_get_outputs
  !   Copy output module variables into C-accessible arguments.
  !   ion arrays have length 11 (nsm-1 where nsm=12).
  ! -------------------------------------------------------------------------
  subroutine c_tglf_get_outputs(              &
       ns_out,                                &
       elec_pflux, elec_eflux, elec_eflux_low,&
       elec_mflux, elec_expwd,                &
       ion_pflux, ion_eflux, ion_eflux_low,   &
       ion_mflux,  ion_expwd                  &
       ) bind(C, name="c_tglf_get_outputs")

    integer(c_int), intent(out) :: ns_out
    real(c_double), intent(out) :: elec_pflux
    real(c_double), intent(out) :: elec_eflux
    real(c_double), intent(out) :: elec_eflux_low
    real(c_double), intent(out) :: elec_mflux
    real(c_double), intent(out) :: elec_expwd
    real(c_double), intent(out) :: ion_pflux(11)
    real(c_double), intent(out) :: ion_eflux(11)
    real(c_double), intent(out) :: ion_eflux_low(11)
    real(c_double), intent(out) :: ion_mflux(11)
    real(c_double), intent(out) :: ion_expwd(11)

    ns_out          = tglf_ns_in
    elec_pflux      = tglf_elec_pflux_out
    elec_eflux      = tglf_elec_eflux_out
    elec_eflux_low  = tglf_elec_eflux_low_out
    elec_mflux      = tglf_elec_mflux_out
    elec_expwd      = tglf_elec_expwd_out
    ion_pflux       = tglf_ion_pflux_out
    ion_eflux       = tglf_ion_eflux_out
    ion_eflux_low   = tglf_ion_eflux_low_out
    ion_mflux       = tglf_ion_mflux_out
    ion_expwd       = tglf_ion_expwd_out

  end subroutine c_tglf_get_outputs

end module tglf_c_api
