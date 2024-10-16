! (C) Copyright 2022 United States Government as represented by the Administrator of the National
!     Aeronautics and Space Administration
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

module gsi_covariance_mod

! atlas
use atlas_module,                   only: atlas_fieldset, atlas_field

! fckit
use fckit_mpi_module,               only: fckit_mpi_comm
use fckit_configuration_module,     only: fckit_configuration

! oops
use kinds,                          only: kind_real
use random_mod
use datetime_mod

! saber
use gsi_grid_mod,                   only: gsi_grid

! gsibec
use m_gsibec,                       only: gsibec_init
use m_gsibec,                       only: gsibec_cv_space
use m_gsibec,                       only: gsibec_sv_space
use m_gsibec,                       only: gsibec_befname
use m_gsibec,                       only: gsibec_init_guess
use m_gsibec,                       only: gsibec_set_guess

use guess_grids,                    only: gsiguess_bkgcov_init  ! temporary
use m_rf,                           only: rf_set    ! temporary
use gsi_metguess_mod,               only: gsi_metguess_get
use gsi_bundlemod,                  only: gsi_bundle
use gsi_bundlemod,                  only: gsi_bundlegetpointer

!use m_gsibec,                       only: gsibec_final_guess
!use m_gsibec,                       only: gsibec_final
!use guess_grids,                    only: gsiguess_bkgcov_final ! temporary
!use m_rf,                           only: rf_unset  ! temporary

use control_vectors,                only: control_vector
use control_vectors,                only: cvars2d,cvars3d
use control_vectors,                only: allocate_cv
use control_vectors,                only: deallocate_cv
use control_vectors,                only: assignment(=)

use state_vectors,                  only: allocate_state
use state_vectors,                  only: svars2d,svars3d
use state_vectors,                  only: deallocate_state

use constants,                      only: grav

implicit none
private
public gsi_covariance


! Fortran class header
type :: gsi_covariance
  type(gsi_grid) :: grid
  logical :: bypassGSIbe
  logical :: cv   ! cv=.true.; sv=.false.
  integer :: mp_comm
  integer :: rank
  contains
    procedure, public :: create
    procedure, public :: delete
    procedure, public :: randomize
    procedure, public :: multiply
end type gsi_covariance

character(len=*), parameter :: myname='gsi_covariance_mod'

! --------------------------------------------------------------------------------------------------

contains

! --------------------------------------------------------------------------------------------------

subroutine create(self, comm, config, ntimes, background, firstguess, valid_times)

! Arguments
class(gsi_covariance),     intent(inout) :: self
type(fckit_mpi_comm),      intent(in)    :: comm
type(fckit_configuration), intent(in)    :: config
integer,                   intent(in)    :: ntimes
type(atlas_fieldset), dimension(ntimes), intent(in)    :: background
type(atlas_fieldset), dimension(ntimes), intent(in)    :: firstguess
type(datetime), dimension(ntimes),       intent(in)    :: valid_times

! Locals
character(len=*), parameter :: myname_=myname//'*create'
character(len=:), allocatable :: nml,bef
real(kind=kind_real), pointer :: rank2(:,:)=>NULL()
logical :: bkgmock
integer :: ier,n,itbd,jouter,ii
integer :: ngsivars2d,ngsivars3d
integer, allocatable :: nymd(:), nhms(:)
character(len=20),allocatable :: gsivars(:)
character(len=20),allocatable :: usrvars(:)
character(len=30),allocatable :: tbdvars(:)
character(len=20) :: valid_time_string

! Hold communicator
! -----------------
self%mp_comm=comm%communicator()

! Convert datetime to string in ISO form "yyyy-mm-ddT00:00:00Z"
! -------------------------------------------------------------
allocate(nymd(ntimes),nhms(ntimes))
do ii=1,ntimes
  call datetime_to_string(valid_times(ii), valid_time_string)
  call iso2geos_date_(valid_time_string,nymd(ii),nhms(ii))
enddo

! Create the grid
! ---------------
call self%grid%create(config, comm)
self%rank = comm%rank()

if (.not. self%grid%noGSI) then
  call config%get_or_die("debugging deep bypass gsi B error", self%bypassGSIbe)

! Get required name of resources for GSI B error
! ----------------------------------------------
  call config%get_or_die("gsi berror namelist file",  nml)
  call config%get_or_die("gsi error covariance file", bef)

! Initialize GSI-Berror components
! --------------------------------
  call gsibec_init(self%cv,bkgmock=bkgmock,nmlfile=nml,befile=bef,&
                   layout=self%grid%layout,jouter=jouter,&
                   inymd=nymd,inhms=nhms,&
                   comm=comm%communicator())
  if(jouter==1) call gsibec_init_guess()

! Initialize and set background fields needed by GSI Berror formulation/operators
! -------------------------------------------------------------------------------
  if (bkgmock) then
     call gsiguess_bkgcov_init()
  else
     ! the correct way to handle the guess vars is to get what the met-guess
     ! from GSI vars are, sip through the JEDI firstguess, copy over what is
     ! found and and construct what is missing.

     call gsi_metguess_get('dim::2d',ngsivars2d,ier)
     call gsi_metguess_get('dim::3d',ngsivars3d,ier)
     allocate(tbdvars(ngsivars2d+ngsivars3d))

     ! Inquire about rank-2 vars in GSI met-guess
     do ii=1,ntimes
       itbd=0
       if (ngsivars2d>0) then
           allocate(gsivars(ngsivars2d),usrvars(ngsivars2d))
           call gsi_metguess_get('gsinames::2d',gsivars,ier)
           call gsi_metguess_get('usrnames::2d',usrvars,ier)
           do n=1,ngsivars2d
              itbd=itbd+1
              tbdvars(itbd) = 'unfilled-'//trim(gsivars(n))
              call get_rank2_(rank2,firstguess(ii),trim(usrvars(n)),ier)
              if(ier==0) then
                 call bkg_set2_(trim(gsivars(n)),ii)
                 tbdvars(itbd) = 'filled-'//trim(gsivars(n))
              else
                 tbdvars(itbd) = trim(gsivars(n))
              endif
           enddo
           deallocate(gsivars,usrvars)
       endif
       ! Inquire about rank-3 vars in GSI met-guess
       if (ngsivars3d>0) then
           allocate(gsivars(ngsivars3d),usrvars(ngsivars3d))
           call gsi_metguess_get('gsinames::3d',gsivars,ier)
           call gsi_metguess_get('usrnames::3d',usrvars,ier)
           do n=1,ngsivars3d
              itbd=itbd+1
              tbdvars(itbd) = 'unfilled-'//trim(gsivars(n))
              call get_rank2_(rank2,firstguess(ii),trim(usrvars(n)),ier)
              if(ier==0) then
                 call bkg_set3_(trim(gsivars(n)),ii)
                 tbdvars(itbd) = 'filled-'//trim(gsivars(n))
              else
                 tbdvars(itbd) = trim(gsivars(n))
              endif
           enddo
           deallocate(gsivars,usrvars)
       endif
     enddo ! ntimes

!    Auxiliar vars for GSI-B
!    -----------------------
     call gsiguess_bkgcov_init(tbdvars)
     if (size(tbdvars)>0) then
       if (any(tbdvars(:)(1:6)/='filled')) then
          do n=1,size(tbdvars)
             print *, myname_, ' ', trim(tbdvars(n))
          enddo
          call abor1_ftn(myname_//": missing fields in fg ")
       endif
     endif
     deallocate(tbdvars)

  endif
  if(jouter==1) call rf_set()

endif ! noGSI
deallocate(nymd,nhms)

contains
  subroutine bkg_set2_(varname,islot)

  character(len=*), intent(in) :: varname
  integer,intent(in) :: islot
  real(kind=kind_real), allocatable :: aux(:,:)

! print *, 'Atlas 2-dim: ', size(rank2,2), ' gsi-vec: ', self%grid%lat2,' ', self%grid%lon2
  allocate(aux(self%grid%lat2,self%grid%lon2))
  call addhalo_(rank2(1,:),aux)
  call gsibec_set_guess(varname,islot,aux)
  deallocate(aux)

  end subroutine bkg_set2_
  subroutine bkg_set3_(varname,islot)

  character(len=*), intent(in) :: varname
  integer,intent(in) :: islot
  real(kind=kind_real), allocatable :: aux(:,:,:)

  integer k,npz

! print *, 'Atlas 3-dim: ', size(rank2,2), ' gsi-vec: ', self%grid%lat2,' ', self%grid%lon2
  npz=size(rank2,1)
  allocate(aux(self%grid%lat2,self%grid%lon2,npz))
  if (self%grid%vflip) then
     do k=1,npz
        call addhalo_(rank2(k,:),aux(:,:,npz-k+1))
     enddo
  else
     do k=1,npz
        call addhalo_(rank2(k,:),aux(:,:,k))
     enddo
  endif
  call gsibec_set_guess(varname,islot,aux)
  deallocate(aux)

  end subroutine bkg_set3_

end subroutine create

! --------------------------------------------------------------------------------------------------

subroutine delete(self)

! Arguments
class(gsi_covariance) :: self

! Locals
integer :: ier

! Unfortunately, having these here break the multi-outer-loop opt
! I think the create/delete in saber are working as expected
!if (.not. self%grid%noGSI) then
!  call rf_unset()
!  call gsiguess_bkgcov_final()
!  call gsibec_final_guess()
!  call gsibec_final(.false.)
!endif

! Delete the grid
! ---------------
!call self%grid%delete()

end subroutine delete

! --------------------------------------------------------------------------------------------------

subroutine randomize(self, fields)

! Arguments
class(gsi_covariance), intent(inout) :: self
type(atlas_fieldset),  intent(inout) :: fields

! Locals
type(atlas_field) :: afield
real(kind=kind_real), pointer :: psi(:,:), chi(:,:), t(:,:), q(:,:), qi(:,:), ql(:,:), o3(:,:)
real(kind=kind_real), pointer :: u(:,:), v(:,:)
real(kind=kind_real), pointer :: ps(:,:)

integer, parameter :: rseed = 3

! Get Atlas field
if (fields%has('stream_function').and.fields%has('velocity_potential')) then
  afield = fields%field('stream_function')
  call afield%data(psi)
  afield = fields%field('velocity_potential')
  call afield%data(chi)
elseif (fields%has('eastward_wind').and.fields%has('northward_wind')) then
  afield = fields%field('eastward_wind')
  call afield%data(u)
  afield = fields%field('northward_wind')
  call afield%data(v)
endif

if (fields%has('air_temperature')) then
  afield = fields%field('air_temperature')
  call afield%data(t)
endif

if (fields%has('air_pressure_at_surface')) then
  afield = fields%field('air_pressure_at_surface')
  call afield%data(ps)
endif

if (fields%has('specific_humidity')) then
  afield = fields%field('specific_humidity')
  call afield%data(q)
endif

if (fields%has('cloud_liquid_ice')) then
  afield = fields%field('cloud_liquid_ice')
  call afield%data(qi)
endif

if (fields%has('cloud_liquid_water')) then
  afield = fields%field('cloud_liquid_water')
  call afield%data(ql)
endif

if (fields%has('ozone_mass_mixing_ratio')) then
  afield = fields%field('ozone_mass_mixing_ratio')
  call afield%data(o3)
endif


! Set fields to random numbers
if (associated(psi)) then
  call normal_distribution(psi, 0.0_kind_real, 1.0_kind_real, rseed)
endif
if (associated(u)) then
  call normal_distribution(u, 0.0_kind_real, 1.0_kind_real, rseed)
endif


end subroutine randomize

! --------------------------------------------------------------------------------------------------

subroutine multiply(self, ntimes, fields)

! Arguments
class(gsi_covariance), intent(inout) :: self
integer, intent(in) :: ntimes
type(atlas_fieldset), dimension(ntimes), intent(inout) :: fields

! Locals
character(len=*), parameter :: myname_=myname//'*multiply'
type(atlas_field) :: afield
real(kind=kind_real), pointer :: rank2(:,:)     =>NULL()
real(kind=kind_real), pointer :: gsivar3d(:,:,:)=>NULL()
real(kind=kind_real), pointer :: gsivar2d(:,:)  =>NULL()
real(kind=kind_real), allocatable :: aux(:,:)
real(kind=kind_real), allocatable :: aux1(:)

type(control_vector) :: gsicv
type(gsi_bundle),allocatable :: gsisv(:)
integer :: isc,iec,jsc,jec,npz
integer :: iv,k,ier,itbd,ii

character(len=32),allocatable :: gvars2d(:),gvars3d(:)
character(len=30),allocatable :: tbdvars(:),needvrs(:)

! afield = fields%field('air_pressure_at_surface')
! call afield%data(rank2)
! rank2 = 0.0_kind_real
! rank2(1,int(size(rank1)/2)) = 1.0_kind_real
! return
if (self%grid%noGSI) return

!   gsi-surface: k=1
!   quench-surface: k=1
!   fv3-surface: k=km
isc=self%grid%isc
iec=self%grid%iec
jsc=self%grid%jsc
jec=self%grid%jec
npz=self%grid%npz

! Allocate control vector as defined in GSI
! -----------------------------------------
if (self%cv) then
   allocate(gvars2d(size(cvars2d)),gvars3d(size(cvars3d)))
   gvars2d=cvars2d; gvars3d=cvars3d
   call allocate_cv(gsicv)
else
   allocate(gvars2d(size(svars2d)),gvars3d(size(svars3d)))
   gvars2d=svars2d; gvars3d=svars3d
   allocate(gsisv(ntimes))
   do ii=1,ntimes
      call allocate_state(gsisv(ii),'saber')
   enddo
endif
allocate(tbdvars(size(gvars2d)+size(gvars3d)))
allocate(needvrs(size(gvars2d)+size(gvars3d)))
tbdvars="null"
itbd=0
do iv = 1,size(gvars2d)
   itbd=itbd+1
   tbdvars(itbd) = "unfilled-"//trim(gvars2d(iv))
enddo
do iv = 1,size(gvars3d)
   itbd=itbd+1
   tbdvars(itbd) = "unfilled-"//trim(gvars3d(iv))
enddo

! Convert Atlas fieldsets to GSI bundle fields
! --------------------------------------------
do ii=1,ntimes
   itbd=0
   do iv=1,size(gvars2d)
     itbd=itbd+1
     if (self%cv) then
        call gsi_bundlegetpointer (gsicv%step(ii),gvars2d(iv),gsivar2d,ier)
     else
        call gsi_bundlegetpointer (     gsisv(ii),gvars2d(iv),gsivar2d,ier)
     endif
     if (ier/=0) cycle
     call get_rank2_(rank2,fields(ii),trim(gvars2d(iv)),ier)
     if (ier==0) then
        tbdvars(itbd) = 'filled-'//trim(gvars2d(iv))
     else
        tbdvars(itbd) = trim(gvars2d(iv))
        cycle
     endif
     allocate(aux(size(gsivar2d,1),size(gsivar2d,2)))
     call addhalo_(rank2(1,:),aux)
     gsivar2d=aux
     deallocate(aux)
   enddo
   do iv=1,size(gvars3d)
     itbd=itbd+1
     if (self%cv) then
        call gsi_bundlegetpointer (gsicv%step(ii),gvars3d(iv),gsivar3d,ier)
     else
        call gsi_bundlegetpointer (     gsisv(ii),gvars3d(iv),gsivar3d,ier)
     endif
     if (ier/=0) cycle
     call get_rank2_(rank2,fields(ii),trim(gvars3d(iv)),ier)
     if (ier==0) then
        tbdvars(itbd) = 'filled-'//trim(gvars3d(iv))
     else
        tbdvars(itbd) = trim(gvars3d(iv))
        cycle
     endif
     allocate(aux(size(gsivar3d,1),size(gsivar3d,2)))
     if (self%grid%vflip) then
        do k=1,npz
           call addhalo_(rank2(k,:),aux)
           gsivar3d(:,:,npz-k+1)=aux
        enddo
     else
        do k=1,npz
           call addhalo_(rank2(k,:),aux)
           gsivar3d(:,:,k)=aux
        enddo
     endif
     deallocate(aux)
   enddo
enddo ! ntimes
needvrs=tbdvars

! fill in missing fields
if (self%cv) then
  call cvfix_(gsicv,fields,self%grid%vflip,needvrs,ntimes,'adm')
else
  call svfix_(gsisv,fields,self%grid%vflip,needvrs,ntimes,'adm')
endif
! check that all variables are consistently available
if (any(needvrs(:)(1:6)/='filled')) then
  do iv=1,size(needvrs)
     print *, myname_, '*adm:: ', trim(needvrs(iv))  ! need single PE write/out
  enddo
  call abor1_ftn(myname_//": missing fields in cv(adm) ")
endif

! Apply GSI B-error operator
! --------------------------
if (self%cv) then
   call gsibec_cv_space(gsicv,internalcv=.false.,bypassbe=self%bypassGSIbe)
else
   call gsibec_sv_space(gsisv,internalsv=.false.,bypassbe=self%bypassGSIbe)
endif

! Convert back to Atlas Fields
! ----------------------------
do ii=1,ntimes
   do iv=1,size(gvars2d)
     if (self%cv) then
        call gsi_bundlegetpointer (gsicv%step(ii),gvars2d(iv),gsivar2d,ier)
     else
        call gsi_bundlegetpointer (gsisv(ii),gvars2d(iv),gsivar2d,ier)
     endif
     if (ier/=0) cycle
     call get_rank2_(rank2,fields(ii),trim(gvars2d(iv)),ier)
     if (ier/=0) cycle
     allocate(aux1(size(rank2,2)))
     call remhalo_(gsivar2d,aux1)
     rank2(1,:)=aux1
     deallocate(aux1)
   enddo
   do iv=1,size(gvars3d)
     if (self%cv) then
        call gsi_bundlegetpointer (gsicv%step(ii),gvars3d(iv),gsivar3d,ier)
     else
        call gsi_bundlegetpointer (gsisv(ii),gvars3d(iv),gsivar3d,ier)
     endif
     if (ier/=0) cycle
     call get_rank2_(rank2,fields(ii),trim(gvars3d(iv)),ier)
     if (ier/=0) cycle
     allocate(aux1(size(rank2,2)))
     if (self%grid%vflip) then
        do k=1,npz
           call remhalo_(gsivar3d(:,:,k),aux1)
           rank2(npz-k+1,:)=aux1
        enddo
     else
        do k=1,npz
           call remhalo_(gsivar3d(:,:,k),aux1)
           rank2(k,:)=aux1
        enddo
     endif
     deallocate(aux1)
   enddo
enddo ! ntimes

! Fill in missing fields
needvrs=tbdvars
if (self%cv) then
  call cvfix_(gsicv,fields,self%grid%vflip,needvrs,ntimes,'tlm')
else
  call svfix_(gsisv,fields,self%grid%vflip,needvrs,ntimes,'tlm')
endif
! Check that all variables are consistently available
if (any(needvrs(:)(1:6)/='filled')) then
  do iv=1,size(needvrs)
     print *, myname_, '*tlm:: ', trim(needvrs(iv))  ! need single PE write/out
  enddo
  call abor1_ftn(myname_//": missing fields in cv(tlm) ")
endif


! Release pointer
! ---------------
if (self%cv) then
   call deallocate_cv(gsicv)
else
   do ii=ntimes,1,-1
     call deallocate_state(gsisv(ii))
   enddo
   deallocate(gsisv)
endif
deallocate(needvrs)
deallocate(tbdvars)
deallocate(gvars2d,gvars3d)
call afield%final()


end subroutine multiply

! --------------------------------------------------------------------------------------------------

   subroutine get_rank2_(rank2,fields,vname,ier)
   character(len=*),intent(in):: vname
   type(atlas_fieldset), intent(in):: fields
   real(kind=kind_real), pointer :: rank2(:,:)
   type(atlas_field) :: afield
   integer,intent(out):: ier
   integer,save :: icount = 0
   ier=-1
   if (trim(vname) == 'ps') then
      if (.not.fields%has('air_pressure_at_surface')) return
      afield = fields%field('air_pressure_at_surface')
      call afield%data(rank2)
      ier=0
   endif
!  if (trim(vname) == 'air_pressure_thickness') then
!     if (.not.fields%has('air_pressure_thickness')) return
!     afield = fields%field('air_pressure_thickness')
!     call afield%data(rank2)
!     ier=0
!  endif
   if (trim(vname) == 'ts' .or. trim(vname) == 'sst') then !  ts=gsi background name
      if (.not.fields%has('skin_temperature_at_surface')) return      ! sst=gsi S/CV name
      afield = fields%field('skin_temperature_at_surface')
      call afield%data(rank2)
      icount = icount + 1
      ier=0
   endif
   if (trim(vname) == 'u' .or. trim(vname) == 'ua' ) then
      if (.not.fields%has('eastward_wind')) return
      afield = fields%field('eastward_wind')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'v' .or. trim(vname) == 'va' ) then
      if (.not.fields%has('northward_wind')) return
      afield = fields%field('northward_wind')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'sf') then
      if (.not.fields%has('stream_function')) return
      afield = fields%field('stream_function')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'vp') then
      if (.not.fields%has('velocity_potential')) return
      afield = fields%field('velocity_potential')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 't' .or. trim(vname) == 'tsen' ) then
      if (.not.fields%has('air_temperature')) return
      afield = fields%field('air_temperature')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'tv' ) then
      if (.not.fields%has('virtual_temperature')) return
      afield = fields%field('virtual_temperature')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'q' .or. trim(vname) == 'sphum' ) then
      if (.not.fields%has('specific_humidity')) return
      afield = fields%field('specific_humidity')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'qi') then
      if (.not.fields%has('cloud_liquid_ice')) return
      afield = fields%field('cloud_liquid_ice')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'ql') then
      if (.not.fields%has('cloud_liquid_water')) return
      afield = fields%field('cloud_liquid_water')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'qr') then
      if (.not.fields%has('rain_water')) return
      afield = fields%field('rain_water')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'qs') then
      if (.not.fields%has('snow_water')) return
      afield = fields%field('snow_water')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'qg') then
      if (.not.fields%has('graupel')) return
      afield = fields%field('graupel')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'qh') then
      if (.not.fields%has('hail')) return
      afield = fields%field('hail')
      call afield%data(rank2)
      ier=0
   endif
!  if (trim(vname) == 'cw') then
!     if (.not.fields%has('cloud_water')) return
!     afield = fields%field('cloud_water')
!     call afield%data(rank2)
!     ier=0
!  endif
   if (trim(vname) == 'oz' .or. trim(vname) == 'o3ppmv' ) then
      if (.not.fields%has('mole_fraction_of_ozone_in_air')) return
      afield = fields%field('mole_fraction_of_ozone_in_air')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'o3mr') then
      if (.not.fields%has('ozone_mass_mixing_ratio')) return
      afield = fields%field('ozone_mass_mixing_ratio')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'phis' ) then
      if (.not.fields%has('geopotential_height_times_gravity_at_surface')) then
         if (fields%has('geopotential_height_at_surface')) then
            afield = fields%field('geopotential_height_at_surface')
            call afield%data(rank2)
            rank2 = grav*rank2
            ier=0
         else
            return
         endif
      else
         afield = fields%field('geopotential_height_times_gravity_at_surface')
         call afield%data(rank2)
         ier=0
      end if
   endif
   if (trim(vname) == 'frocean' ) then
      if (.not.fields%has('fraction_of_ocean')) return
       afield = fields%field('fraction_of_ocean')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'frlake' ) then
      if (.not.fields%has('fraction_of_lake')) return
      afield = fields%field('fraction_of_lake')
      call afield%data(rank2)
      ier=0
   endif
   if (trim(vname) == 'frseaice' ) then
      if (.not.fields%has('fraction_of_ice')) return
      afield = fields%field('fraction_of_ice')
      call afield%data(rank2)
      ier=0
   endif
   end subroutine get_rank2_

   subroutine addhalo_(rank,var)
   real(kind=kind_real),intent(in) :: rank(:)
   real(kind=kind_real),intent(inout):: var(:,:)
   integer ii,jj,jnode
   integer mylat2,mylon2,ndim
   mylat2 = size(var,1)
   mylon2 = size(var,2)
   ndim = (mylat2-2)*(mylon2-2)
   jnode=1
   var = sum(rank(jnode:ndim))/ndim ! RT_TBD: hack to fill in halo
   do jj=2,mylat2-1
      do ii=2,mylon2-1
         var(jj,ii) = rank(jnode)
         jnode = jnode + 1
      enddo
   enddo
   end subroutine addhalo_

   subroutine remhalo_(var,rank)
   real(kind=kind_real),intent(in) :: var(:,:)
   real(kind=kind_real),intent(out):: rank(:)
   integer ii,jj,jnode
   integer mylat2,mylon2
   mylat2 = size(var,1)
   mylon2 = size(var,2)
   jnode=1
   do jj=2,mylat2-1
      do ii=2,mylon2-1
         rank(jnode) = var(jj,ii)
         jnode = jnode + 1
      enddo
   enddo
   end subroutine remhalo_

   subroutine cvfix_(gsicv,jedicv,vflip,need,ntimes,which)

   use control_vectors, only: control_vector
   use gsi_bundlemod, only: gsi_bundlegetpointer
   use gsi_metguess_mod, only: gsi_metguess_bundle
   use gsi_metguess_mod, only: gsi_metguess_get
   use gsi_convert_cv_mod, only: gsi_tv_to_t_tl
   use gsi_convert_cv_mod, only: gsi_tv_to_t_ad

   implicit none

   type(control_vector),intent(inout) :: gsicv
   type(atlas_fieldset),intent(inout) :: jedicv(:)
   logical,intent(in) :: vflip
   integer,intent(in) :: ntimes
   character(len=*),intent(inout) :: need(:)
   character(len=*),intent(in) :: which
!
   real(kind=kind_real), allocatable :: t_pt(:,:,:)
   real(kind=kind_real), pointer ::    tv(:,:,:)=>NULL()
   real(kind=kind_real), pointer :: tv_pt(:,:,:)=>NULL()
   real(kind=kind_real), pointer ::     q(:,:,:)=>NULL()
   real(kind=kind_real), pointer ::  q_pt(:,:,:)=>NULL()
   real(kind=kind_real), pointer ::   rank2(:,:)=>NULL()
   real(kind=kind_real), pointer ::     sst(:,:)=>NULL()
   real(kind=kind_real), allocatable :: aux1(:)
   integer k,npz,ii,ier
!
   if(size(need)<1) return
   if (any(need=='tv')) then
     do ii=1,ntimes
      ! from first guess ...
      call gsi_bundlegetpointer(gsi_metguess_bundle(ii),'q' ,q ,ier)
      call gsi_bundlegetpointer(gsi_metguess_bundle(ii),'tv',tv,ier)
      ! from GSI cv ...
      call gsi_bundlegetpointer(gsicv%step(ii),'q' ,q_pt ,ier)
      call gsi_bundlegetpointer(gsicv%step(ii),'tv',tv_pt,ier)
      ! from JEDI cv ...
      call get_rank2_(rank2,jedicv(ii),'t',ier)
      npz=size(q,3)
      allocate(t_pt(size(q,1),size(q,2),size(q,3)))
      if (vflip) then
         do k=1,npz
            call addhalo_(rank2(k,:),t_pt(:,:,npz-k+1))
         enddo
      else
         do k=1,npz
            call addhalo_(rank2(k,:),t_pt(:,:,k))
         enddo
      endif
      ! retrieve missing field
      if(which=='tlm') then
        call gsi_tv_to_t_tl(tv,tv_pt,q,q_pt,t_pt)
        ! pass it back to JEDI ...
        allocate(aux1(size(rank2,2)))
        if (vflip) then
           do k=1,npz
              call remhalo_(t_pt(:,:,k),aux1)
              rank2(npz-k+1,:)=aux1
           enddo
        else
           do k=1,npz
              call remhalo_(t_pt(:,:,k),aux1)
              rank2(k,:)=aux1
           enddo
        endif
        deallocate(aux1)
        where(need=='tv')
           need='filled-'//need
        endwhere
      endif
      if(which=='adm') then
        call gsi_tv_to_t_ad(tv,tv_pt,q,q_pt,t_pt)
        where(need=='tv')
           need='filled-'//need
        endwhere
      endif
      deallocate(t_pt)
     enddo
   endif
!  SST GSI-way
   if (any(need=='ts')) then
     do ii=1,ntimes
      ! from JEDI cv ...
      call get_rank2_(rank2,jedicv(ii),'ts',ier)
      ! from GSI cv ...
      call gsi_bundlegetpointer(gsicv%step(ii),'sst' ,sst ,ier)
      if(which=='tlm') then
         allocate(aux1(size(rank2,2)))
         call remhalo_(sst,aux1)
         if (vflip) then
            rank2(1,:)   = aux1
         else
            rank2(npz,:) = aux1
         endif
         deallocate(aux1)
      endif
      if(which=='adm') then
         if (vflip) then
            call addhalo_(rank2(1,:),sst)
         else
            call addhalo_(rank2(npz,:),sst)
         endif
      endif
      where(need=='sst')
        need='filled-'//need
      endwhere
     enddo
   endif

   end subroutine cvfix_

   subroutine svfix_(gsisv,jedicv,vflip,need,ntimes,which)

   use gsi_bundlemod, only: gsi_bundle
   use gsi_bundlemod, only: gsi_bundlegetpointer
   use gsi_metguess_mod, only: gsi_metguess_bundle
   use gsi_metguess_mod, only: gsi_metguess_get
   use gsi_convert_cv_mod, only: gsi_tv_to_t_tl
   use gsi_convert_cv_mod, only: gsi_tv_to_t_ad

   implicit none

   character, parameter :: myname_ = myname//'*svfix_'
   type(gsi_bundle),intent(inout) :: gsisv(:)
   type(atlas_fieldset),intent(inout) :: jedicv(:)
   logical,intent(in) :: vflip
   integer,intent(in) :: ntimes
   character(len=*),intent(inout) :: need(:)
   character(len=*),intent(in) :: which
!
   real(kind=kind_real), allocatable :: t_pt(:,:,:)
   real(kind=kind_real), pointer ::       tv(:,:,:)=>NULL()
   real(kind=kind_real), pointer ::    tv_pt(:,:,:)=>NULL()
   real(kind=kind_real), pointer ::        q(:,:,:)=>NULL()
   real(kind=kind_real), pointer ::     q_pt(:,:,:)=>NULL()
   real(kind=kind_real), pointer ::      rank2(:,:)=>NULL()
   real(kind=kind_real), allocatable :: aux1(:)
   integer k,npz,ii,ier
!
   if(size(need)<1) return

   if (any(need=='prse')) then
       where(need=='prse')  ! gsi will take care of this
          need='filled-'//need
       endwhere
   endif

   if (any(need=='tv')) then
     do ii=1,ntimes
      ! from first guess ...
      call gsi_bundlegetpointer(gsi_metguess_bundle(ii),'q' ,q ,ier)
      call gsi_bundlegetpointer(gsi_metguess_bundle(ii),'tv',tv,ier)
      ! from GSI cv ...
      call gsi_bundlegetpointer(gsisv(ii),'q' ,q_pt ,ier)
      call gsi_bundlegetpointer(gsisv(ii),'tv',tv_pt,ier)
      ! from JEDI cv ...
      call get_rank2_(rank2,jedicv(ii),'t',ier)
      npz=size(q,3)
      allocate(t_pt(size(q,1),size(q,2),size(q,3)))
      if (vflip) then
         do k=1,npz
            call addhalo_(rank2(k,:),t_pt(:,:,npz-k+1))
         enddo
      else
         do k=1,npz
            call addhalo_(rank2(k,:),t_pt(:,:,k))
         enddo
      endif
      ! retrieve missing field
      if(which=='tlm') then
        call gsi_tv_to_t_tl(tv,tv_pt,q,q_pt,t_pt)
        where(need=='tv')
           need='filled-'//need
        endwhere
        ! pass it back to JEDI ...
        allocate(aux1(size(rank2,2)))
        if (vflip) then
           do k=1,npz
              call remhalo_(t_pt(:,:,k),aux1)
              rank2(npz-k+1,:)=aux1
           enddo
        else
           do k=1,npz
              call remhalo_(t_pt(:,:,k),aux1)
              rank2(k,:)=aux1
           enddo
        endif
        deallocate(aux1)
      endif
      if(which=='adm') then
        call gsi_tv_to_t_ad(tv,tv_pt,q,q_pt,t_pt)
        where(need=='tv')
           need='filled-'//need
        endwhere
      endif
      deallocate(t_pt)
     enddo
   endif
   end subroutine svfix_

   subroutine iso2geos_date_(str,nymd,nhms)
   implicit none
   character(len=*), intent(in) :: str
   integer, intent(out) :: nymd,nhms
   integer yyyy,mm,dd
   integer hh,mn,ss
   read(str(1: 4),*) yyyy
   read(str(6: 7),*) mm
   read(str(9:10),*) dd
   nymd = 10000*yyyy + 100*mm + dd
   read(str(12:13),*) hh
   read(str(15:16),*) mn
   read(str(18:19),*) ss
   nhms = 10000*hh + 100*mn + ss
   end subroutine iso2geos_date_

end module gsi_covariance_mod
