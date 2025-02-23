import copy
import numpy as np
from collections import OrderedDict

try:
    import xarray as xr
except:
    pass
try:
    from IPython import embed
except:
    pass

"""
This set of routines is used to create a Plasmastate class by reading a standard
plasmastate file (.cdf) that has been generated using TRXPL.

The method modify_default applied to the class:
	1) 	looks for charge states of impurities and lumps them all together by computing
		a volume-average charge and subsequently computing impurity density to satisfy
		quasineutrality and Zeff
	2) 	looks for fusion ions other than He4 (i.e. fast fusion H, T and He3) and removes
		them from the file

The resulting plasmastate file can then be used by profiles_gen without problems with
"Too many ions".
"""

from mitim_tools.misc_tools.LOGtools import printMsg as print

class Plasmastate:
    def __init__(self, CDFfile):
        self.CDFfile = CDFfile

        # ------------------------------------------------------------------
        #       Reading Original PLASMASTATE
        # ------------------------------------------------------------------

        self.plasma_orig = xr.open_dataset(self.CDFfile)
        self.plasma = copy.deepcopy(self.plasma_orig)

    def modify_default(
        self,
        CDFfile_new,
        shotNumber=12345,
        RemoveFusionIons=["He3_fusn", "H_fusn", "T_fusn"],
        RemoveTHERMALIons=None,
    ):

        if RemoveTHERMALIons is None:
            RemoveTHERMALIons = []

        self.CDFfile_new = CDFfile_new

        print(f"\t- Modifying {self.CDFfile} Plasmastate file...")

        self.lumpChargeStates(self.CDFfile_new + "_1")
        try:
            self.removeExtraFusionIons(
                self.CDFfile_new + "_2", speciesNames=RemoveFusionIons
            )
        except:
            print(
                " --> I could not remove extra fusion ions. Probably because TRANSP was run without fusion reactions",
            )
        self.removeExtraTHERMALIons(
            self.CDFfile_new + "_3", speciesNames=RemoveTHERMALIons
        )

        self.addShotNumber(self.CDFfile_new, shotNumber)

        print(
            "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        )
        print(
            f"{self.CDFfile} Plasmastate modified successfully",
        )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------

    def lumpChargeStates(self, fileNew):
        # ------------------------------------------------------------------
        #       Creating new Dataset
        # ------------------------------------------------------------------

        impNames = []
        for i in self.plasma["SIMPI_name"].values:
            try:
                ii = i.strip().decode()
            except:
                ii = i.strip()
            impNames.append(ii.split("_")[0])
        impNamesUnique = np.unique(impNames)

        # In the all-impurities array given by 'nspec_impi', group all charge states together

        DimensionsVariables = OrderedDict()
        DimensionsVariables["dim_nspec_impi"] = "SIMPI_name"
        DimensionsVariables["dp1_nspec_all"] = "ALL_name"
        DimensionsVariables["dp1_nspec_th"] = "S_name"

        for namesDims in DimensionsVariables:
            print(
                f"\n~~~~~~~~~ Working with dimension {namesDims}",
            )

            for impName in impNamesUnique:
                # Define density and charge vectors for all charge states of this impurity

                Zave_mean, nZave = defineLumpingVectors(self.plasma, impName)

                if Zave_mean is not None:
                    startLump, endLump = findIndecesChargeStates(
                        self.plasma[DimensionsVariables[namesDims]].values, impName
                    )

                    NewData = lumpImpurities_Dictionary(
                        self.plasma,
                        namesDims,
                        Zave_mean,
                        nZave,
                        startLump=startLump,
                        endLump=endLump,
                        nameImp=impName,
                    )
                    self.plasma = xr.Dataset(NewData)

                else:
                    print(
                        f"Charge States for impurity {impName} could not be found...",
                    )

        # ------------------------------------------------------------------
        #       Writting New PLASMASTATE
        # ------------------------------------------------------------------

        self.plasma.to_netcdf(fileNew, format="NETCDF3_CLASSIC")

        # ------------------------------------------------------------------
        #       Reading New PLASMASTATE
        # ------------------------------------------------------------------

        self.plasma = xr.open_dataset(fileNew)
        self.CDFfile = fileNew

    def removeExtraTHERMALIons(self, fileNew, speciesNames=["H"]):
        # ------------------------------------------------------------------
        #       Creating new Dataset
        # ------------------------------------------------------------------

        # In the all-impurities array given by 'nspec_impi', group all charge states together

        DimensionsVariables = OrderedDict(
            {
                "dp1_nspec_alla": "ALLA_name",
                "dp1_nspec_th": "S_name",
                "dp1_nspec_all": "ALL_name",
            }
        )

        for namesDims in DimensionsVariables:
            print(
                f"\n\n~~~~~~~~~ Working with dimension {namesDims}",
            )

            for fastName in speciesNames:
                index = None
                # Find index
                for cont, i in enumerate(
                    self.plasma[DimensionsVariables[namesDims]].values
                ):
                    try:
                        j = i.strip().decode()
                    except:
                        j = i.strip()
                    if j == fastName:
                        index = cont

                if index is not None:
                    NewData = removeSpecies_Dictionary(
                        self.plasma, namesDims, index=index, nameVar=fastName
                    )
                    self.plasma = xr.Dataset(NewData)

                else:
                    print(
                        f"Fusion ion {fastName} could not be found...",
                    )

        # ------------------------------------------------------------------
        #       Writting New PLASMASTATE
        # ------------------------------------------------------------------

        self.plasma.to_netcdf(fileNew, format="NETCDF3_CLASSIC")

        # ------------------------------------------------------------------
        #       Reading New PLASMASTATE
        # ------------------------------------------------------------------

        self.plasma = xr.open_dataset(fileNew)

    def removeExtraFusionIons(
        self, fileNew, speciesNames=["He3_fusn", "H_fusn", "T_fusn"]
    ):
        # ------------------------------------------------------------------
        #       Creating new Dataset
        # ------------------------------------------------------------------

        # In the all-impurities array given by 'nspec_impi', group all charge states together

        DimensionsVariables = OrderedDict(
            {
                "dp1_nspec_alla": "ALLA_name",
                "dim_nspec_fusion": "SFUS_name",
                "dp1_nspec_all": "ALL_name",
            }
        )

        for namesDims in DimensionsVariables:
            print(
                f"\n\n~~~~~~~~~ Working with dimension {namesDims}",
            )

            for fastName in speciesNames:
                index = None
                # Find index
                for cont, i in enumerate(
                    self.plasma[DimensionsVariables[namesDims]].values
                ):
                    try:
                        j = i.strip().decode()
                    except:
                        j = i.strip()
                    if j == fastName:
                        index = cont

                if index is not None:
                    NewData = removeSpecies_Dictionary(
                        self.plasma, namesDims, index=index, nameVar=fastName
                    )
                    self.plasma = xr.Dataset(NewData)

                else:
                    print(
                        f"Fusion ion {fastName} could not be found...",
                    )

        # ------------------------------------------------------------------
        #       Writting New PLASMASTATE
        # ------------------------------------------------------------------

        self.plasma.to_netcdf(fileNew, format="NETCDF3_CLASSIC")

        # ------------------------------------------------------------------
        #       Reading New PLASMASTATE
        # ------------------------------------------------------------------

        self.plasma = xr.open_dataset(fileNew)

    def addShotNumber(self, fileNew, num):
        print(
            f"\n\n~~~~~~~~~ Changing shot number specification to {num}",
        )

        # ------------------------------------------------------------------
        #       Creating new Dataset
        # ------------------------------------------------------------------

        self.plasma["shot_number"] = (
            self.plasma["shot_number"].dims,
            str(num),
            self.plasma["shot_number"].attrs,
        )

        NewData = {}
        for ikey in self.plasma.keys():
            if "shot_number" in ikey:
                NewData[ikey] = (self.plasma[ikey].dims, num, self.plasma[ikey].attrs)
            else:
                NewData[ikey] = (
                    self.plasma[ikey].dims,
                    self.plasma[ikey].data,
                    self.plasma[ikey].attrs,
                )

        self.plasma = xr.Dataset(NewData)

        # ------------------------------------------------------------------
        #       Writting New PLASMASTATE
        # ------------------------------------------------------------------

        self.plasma.to_netcdf(fileNew, format="NETCDF3_CLASSIC")

        # ------------------------------------------------------------------
        #       Reading New PLASMASTATE
        # ------------------------------------------------------------------

        self.plasma = xr.open_dataset(fileNew)
        self.CDFfile = fileNew

    def plot(self, axs=None, color="r", label=""):
        if axs is None:
            provided = False
            plt.ion()
            fig, axs = plt.subplots(ncols=3)
        else:
            provided = True

        # Rho definition
        rho = self.plasma["rho"].values[:-1]
        rho = rho + (rho[1] - rho[0]) / 2

        TekeV = self.plasma["Ts"].values[0, :]
        TikeV = self.plasma["Ti"].values
        ne20 = self.plasma["ns"].values[0, :] * 1e-20

        ax = axs[0]
        ax.plot(rho, TekeV, c=color, label=label)
        if not provided:
            ax.set_xlabel("rho")
            ax.set_ylabel("Te (keV)")

        ax = axs[1]
        ax.plot(rho, TikeV, c=color, label=label)
        if not provided:
            ax.set_xlabel("rho")
            ax.set_ylabel("Ti (keV)")

        ax = axs[2]
        ax.plot(rho, ne20, c=color, label=label)
        if not provided:
            ax.set_xlabel("rho")
            ax.set_ylabel("ne (1E20m^-3)")


# --------------------------------------------------------------------------------------


def lumpingMetric(z_orig, Zave_mean, nZave, nameVar):
    """
    This is where the lumping takes place, z_orig has (num_species,num_rho) dimensions and
    must be converted into a 1D array (a single specie)
    """

    # -----------------------------------------------
    # Operate on variables
    # -----------------------------------------------

    if nameVar in ["q_S", "q_ALL"]:
        z = Zave_mean
    # If the variable is not density nor change, simply compute the weighted average with density
    elif nameVar in ["ns"]:
        z = nZave
    # If the variable is not density nor change, simply compute the average
    else:
        z = np.mean(z_orig, axis=0)

    # # If the variable is not density nor change, simply compute the weighted average with density
    # else:						z = np.mean(z_orig*nZ_all,axis=0) / np.mean(nZ_all,axis=0)

    return z


def removeSpecies_Dictionary(DataSet_orig, DimensionToChange, index=100, nameVar="MITIM"):
    print(
        "\nOriginal species dimension: {0}, removing {1} in index {2}".format(
            DataSet_orig.dims[DimensionToChange], nameVar, index
        ),
    )

    NewData = {}
    for ikey in DataSet_orig.keys():
        if DimensionToChange in DataSet_orig[ikey].dims:
            NewData[ikey] = change_removeVariable(ikey, DataSet_orig[ikey], index)
        else:
            NewData[ikey] = (
                DataSet_orig[ikey].dims,
                DataSet_orig[ikey].data,
                DataSet_orig[ikey].attrs,
            )

    return NewData


def change_removeVariable(name, varOrig, index):
    print(f"Changing \"{varOrig.attrs['long_name']}\"")

    newData = removeVariable(varOrig.data, index)
    NewDims = varOrig.dims

    varNew = (NewDims, newData, varOrig.attrs)

    return varNew


def removeVariable(z_orig, index):
    z = []
    WasIncluded = False
    for i in range(z_orig.shape[0]):
        if i != index:
            z.append(z_orig[i])

    z = np.array(z)

    return z


def lumpImpurities_Dictionary(
    DataSet_orig,
    DimensionToChange,
    Zave_mean,
    nZave,
    startLump=1,
    endLump=100,
    nameImp="MITIM",
):
    print(
        "\nOriginal species dimension: {0}, Lumping from {1} to {2}".format(
            DataSet_orig.dims[DimensionToChange], startLump, endLump
        ),
    )

    NewData = {}
    for ikey in DataSet_orig.keys():
        if DimensionToChange in DataSet_orig[ikey].dims:
            NewData[ikey] = changeVariable_Lump(
                ikey,
                DataSet_orig[ikey],
                startLump,
                endLump,
                Zave_mean,
                nZave,
                nameImp=nameImp,
            )
        else:
            NewData[ikey] = (
                DataSet_orig[ikey].dims,
                DataSet_orig[ikey].data,
                DataSet_orig[ikey].attrs,
            )

    return NewData


def changeVariable_Lump(
    name, varOrig, startLump, endLump, Zave_mean, nZave, nameImp="MITIM"
):
    print(
        'Changing "{0}" because dimensions {1} for {2}'.format(
            varOrig.attrs["long_name"], varOrig.dims, name
        ),
    )

    newData = lumpVariable(
        varOrig.data,
        startLump,
        endLump,
        Zave_mean,
        nZave,
        IsItName="_name" in name,
        nameImp=nameImp,
        nameVar=name,
    )

    # if len(newData.shape) == 1: 	NewDims = (u'dp1_nspec_th1')
    # else: 							NewDims = (u'dp1_nspec_th1', u'dm1_nrho')
    NewDims = varOrig.dims

    varNew = (NewDims, newData, varOrig.attrs)

    return varNew


def lumpVariable(
    z_orig,
    startLump,
    endLump,
    Zave_mean,
    nZave,
    IsItName=False,
    nameImp="MITIM",
    nameVar="",
):
    z = []
    WasIncluded = False
    for i in range(z_orig.shape[0]):
        if i < startLump or i > endLump:
            z.append(z_orig[i])
        elif not WasIncluded:
            if IsItName:
                z.append(nameImp)
            else:
                z.append(
                    lumpingMetric(z_orig[startLump:endLump], Zave_mean, nZave, nameVar)
                )
            WasIncluded = True

    z = np.array(z)

    return z


def defineLumpingVectors(d, impName):
    startLump, endLump = findIndecesChargeStates(d["S_name"].values, impName)

    if startLump is not None:
        nZ_all = d["ns"].values[startLump : endLump + 1]
        qZ_all = d["q_S"].values[startLump : endLump + 1]
        volZone = np.diff(d["vol"].values)

        # Convert qZ_all to 2D
        qZ_all = np.transpose(np.tile(qZ_all, (nZ_all.shape[1], 1)))

        # Compute SUM(n*Z) and SUM(n*Z^2)
        nZsum = np.sum(nZ_all * qZ_all, axis=0)
        nZ2sum = np.sum(nZ_all * qZ_all**2, axis=0)

        # Calculate volume-average of impurity charge
        Zave = nZ2sum / nZsum
        Zave_mean = np.mean(Zave * volZone) / np.mean(volZone)
        print(f"Zave = {Zave_mean / 1.6022e-19}")

        # Calculate the average impurity density
        nZave = nZsum / Zave_mean

        return Zave_mean, nZave

    else:
        return None, None


def findIndecesChargeStates(namesSearch, impName):
    startLump = None
    # Find starting index for impurity
    for cont, i in enumerate(namesSearch):
        try:
            j = i.strip().decode()
        except:
            j = i.strip()
        if j == impName + "_1":
            startLump = cont

    if startLump is not None:
        # Find the indeces of all charge states for that impurity
        for cont, i in enumerate(namesSearch):
            try:
                j = i.decode()
            except:
                j = i
            if impName not in j and cont > startLump:
                endLump = cont - 1
                break  # This would not work if impurities share names in the form: 'F' and 'Fe'
            else:
                endLump = cont

        return startLump, endLump

    else:
        return None, None
