/*
   There are a total of 200 entries in the algebraic variable array.
   There are a total of 49 entries in each of the rate and state variable arrays.
   There are a total of 206 entries in the constant variable array.
 */

#include "Ohara_Rudy_cipa_v1_2017.hpp"
#include <cmath>
#include <cstdlib>
// #include "../../functions/inputoutput.hpp"
#include <cstdio>
#include "../modules/glob_funct.hpp"
#include <cuda_runtime.h>
#include <cuda.h>

/*
 * TIME is time in component environment (millisecond).
 * CONSTANTS[celltype] is celltype in component environment (dimensionless).
 * CONSTANTS[nao] is nao in component extracellular (millimolar).
 * CONSTANTS[cao] is cao in component extracellular (millimolar).
 * CONSTANTS[ko] is ko in component extracellular (millimolar).
 * CONSTANTS[R] is R in component physical_constants (joule_per_kilomole_kelvin).
 * CONSTANTS[T] is T in component physical_constants (kelvin).
 * CONSTANTS[F] is F in component physical_constants (coulomb_per_mole).
 * CONSTANTS[zna] is zna in component physical_constants (dimensionless).
 * CONSTANTS[zca] is zca in component physical_constants (dimensionless).
 * CONSTANTS[zk] is zk in component physical_constants (dimensionless).
 * CONSTANTS[L] is L in component cell_geometry (centimeter).
 * CONSTANTS[rad] is rad in component cell_geometry (centimeter).
 * CONSTANTS[vcell] is vcell in component cell_geometry (microliter).
 * CONSTANTS[Ageo] is Ageo in component cell_geometry (centimeter_squared).
 * CONSTANTS[Acap] is Acap in component cell_geometry (centimeter_squared).
 * CONSTANTS[vmyo] is vmyo in component cell_geometry (microliter).
 * CONSTANTS[vnsr] is vnsr in component cell_geometry (microliter).
 * CONSTANTS[vjsr] is vjsr in component cell_geometry (microliter).
 * CONSTANTS[vss] is vss in component cell_geometry (microliter).
 * STATES[V] is v in component membrane (millivolt).
 * ALGEBRAIC[vfrt] is vfrt in component membrane (dimensionless).
 * CONSTANTS[ffrt] is ffrt in component membrane (coulomb_per_mole_millivolt).
 * CONSTANTS[frt] is frt in component membrane (per_millivolt).
 * ALGEBRAIC[INa] is INa in component INa (microA_per_microF).
 * ALGEBRAIC[INaL] is INaL in component INaL (microA_per_microF).
 * ALGEBRAIC[Ito] is Ito in component Ito (microA_per_microF).
 * ALGEBRAIC[ICaL] is ICaL in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaNa] is ICaNa in component ICaL (microA_per_microF).
 * ALGEBRAIC[ICaK] is ICaK in component ICaL (microA_per_microF).
 * ALGEBRAIC[IKr] is IKr in component IKr (microA_per_microF).
 * ALGEBRAIC[IKs] is IKs in component IKs (microA_per_microF).
 * ALGEBRAIC[IK1] is IK1 in component IK1 (microA_per_microF).
 * ALGEBRAIC[INaCa_i] is INaCa_i in component INaCa_i (microA_per_microF).
 * ALGEBRAIC[INaCa_ss] is INaCa_ss in component INaCa_i (microA_per_microF).
 * ALGEBRAIC[INaK] is INaK in component INaK (microA_per_microF).
 * ALGEBRAIC[INab] is INab in component INab (microA_per_microF).
 * ALGEBRAIC[IKb] is IKb in component IKb (microA_per_microF).
 * ALGEBRAIC[IpCa] is IpCa in component IpCa (microA_per_microF).
 * ALGEBRAIC[ICab] is ICab in component ICab (microA_per_microF).
 * ALGEBRAIC[Istim] is Istim in component membrane (microA_per_microF).
 * CONSTANTS[stim_start] is stim_start in component membrane (millisecond).
 * CONSTANTS[stim_end] is stim_end in component membrane (millisecond).
 * CONSTANTS[amp] is amp in component membrane (microA_per_microF).
 * CONSTANTS[BCL] is BCL in component membrane (millisecond).
 * CONSTANTS[duration] is duration in component membrane (millisecond).
 * CONSTANTS[KmCaMK] is KmCaMK in component CaMK (millimolar).
 * CONSTANTS[aCaMK] is aCaMK in component CaMK (per_millimolar_per_millisecond).
 * CONSTANTS[bCaMK] is bCaMK in component CaMK (per_millisecond).
 * CONSTANTS[CaMKo] is CaMKo in component CaMK (dimensionless).
 * CONSTANTS[KmCaM] is KmCaM in component CaMK (millimolar).
 * ALGEBRAIC[CaMKb] is CaMKb in component CaMK (millimolar).
 * ALGEBRAIC[CaMKa] is CaMKa in component CaMK (millimolar).
 * STATES[CaMKt] is CaMKt in component CaMK (millimolar).
 * STATES[cass] is cass in component intracellular_ions (millimolar).
 * CONSTANTS[cmdnmax_b] is cmdnmax_b in component intracellular_ions (millimolar).
 * CONSTANTS[cmdnmax] is cmdnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmcmdn] is kmcmdn in component intracellular_ions (millimolar).
 * CONSTANTS[trpnmax] is trpnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmtrpn] is kmtrpn in component intracellular_ions (millimolar).
 * CONSTANTS[BSRmax] is BSRmax in component intracellular_ions (millimolar).
 * CONSTANTS[KmBSR] is KmBSR in component intracellular_ions (millimolar).
 * CONSTANTS[BSLmax] is BSLmax in component intracellular_ions (millimolar).
 * CONSTANTS[KmBSL] is KmBSL in component intracellular_ions (millimolar).
 * CONSTANTS[csqnmax] is csqnmax in component intracellular_ions (millimolar).
 * CONSTANTS[kmcsqn] is kmcsqn in component intracellular_ions (millimolar).
 * STATES[nai] is nai in component intracellular_ions (millimolar).
 * STATES[nass] is nass in component intracellular_ions (millimolar).
 * STATES[ki] is ki in component intracellular_ions (millimolar).
 * STATES[kss] is kss in component intracellular_ions (millimolar).
 * STATES[cansr] is cansr in component intracellular_ions (millimolar).
 * STATES[cajsr] is cajsr in component intracellular_ions (millimolar).
 * STATES[cai] is cai in component intracellular_ions (millimolar).
 * ALGEBRAIC[JdiffNa] is JdiffNa in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jdiff] is Jdiff in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jup] is Jup in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[JdiffK] is JdiffK in component diff (millimolar_per_millisecond).
 * ALGEBRAIC[Jrel] is Jrel in component ryr (millimolar_per_millisecond).
 * ALGEBRAIC[Jtr] is Jtr in component trans_flux (millimolar_per_millisecond).
 * ALGEBRAIC[Bcai] is Bcai in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcajsr] is Bcajsr in component intracellular_ions (dimensionless).
 * ALGEBRAIC[Bcass] is Bcass in component intracellular_ions (dimensionless).
 * CONSTANTS[cm] is cm in component intracellular_ions (microF_per_centimeter_squared).
 * CONSTANTS[PKNa] is PKNa in component reversal_potentials (dimensionless).
 * ALGEBRAIC[ENa] is ENa in component reversal_potentials (millivolt).
 * ALGEBRAIC[EK] is EK in component reversal_potentials (millivolt).
 * ALGEBRAIC[EKs] is EKs in component reversal_potentials (millivolt).
 * ALGEBRAIC[mss] is mss in component INa (dimensionless).
 * ALGEBRAIC[tm] is tm in component INa (millisecond).
 * CONSTANTS[mssV1] is mssV1 in component INa (millivolt).
 * CONSTANTS[mssV2] is mssV2 in component INa (millivolt).
 * CONSTANTS[mtV1] is mtV1 in component INa (millivolt).
 * CONSTANTS[mtV2] is mtV2 in component INa (millivolt).
 * CONSTANTS[mtD1] is mtD1 in component INa (dimensionless).
 * CONSTANTS[mtD2] is mtD2 in component INa (dimensionless).
 * CONSTANTS[mtV3] is mtV3 in component INa (millivolt).
 * CONSTANTS[mtV4] is mtV4 in component INa (millivolt).
 * STATES[m] is m in component INa (dimensionless).
 * ALGEBRAIC[hss] is hss in component INa (dimensionless).
 * ALGEBRAIC[thf] is thf in component INa (millisecond).
 * ALGEBRAIC[ths] is ths in component INa (millisecond).
 * CONSTANTS[hssV1] is hssV1 in component INa (millivolt).
 * CONSTANTS[hssV2] is hssV2 in component INa (millivolt).
 * CONSTANTS[Ahs] is Ahs in component INa (dimensionless).
 * CONSTANTS[Ahf] is Ahf in component INa (dimensionless).
 * STATES[hf] is hf in component INa (dimensionless).
 * STATES[hs] is hs in component INa (dimensionless).
 * ALGEBRAIC[h] is h in component INa (dimensionless).
 * CONSTANTS[GNa] is GNa in component INa (milliS_per_microF).
 * CONSTANTS[shift_INa_inact] is shift_INa_inact in component INa (millivolt).
 * ALGEBRAIC[jss] is jss in component INa (dimensionless).
 * ALGEBRAIC[tj] is tj in component INa (millisecond).
 * STATES[j] is j in component INa (dimensionless).
 * ALGEBRAIC[hssp] is hssp in component INa (dimensionless).
 * ALGEBRAIC[thsp] is thsp in component INa (millisecond).
 * STATES[hsp] is hsp in component INa (dimensionless).
 * ALGEBRAIC[hp] is hp in component INa (dimensionless).
 * ALGEBRAIC[tjp] is tjp in component INa (millisecond).
 * STATES[jp] is jp in component INa (dimensionless).
 * ALGEBRAIC[fINap] is fINap in component INa (dimensionless).
 * ALGEBRAIC[mLss] is mLss in component INaL (dimensionless).
 * ALGEBRAIC[tmL] is tmL in component INaL (millisecond).
 * STATES[mL] is mL in component INaL (dimensionless).
 * CONSTANTS[thL] is thL in component INaL (millisecond).
 * ALGEBRAIC[hLss] is hLss in component INaL (dimensionless).
 * STATES[hL] is hL in component INaL (dimensionless).
 * ALGEBRAIC[hLssp] is hLssp in component INaL (dimensionless).
 * CONSTANTS[thLp] is thLp in component INaL (millisecond).
 * STATES[hLp] is hLp in component INaL (dimensionless).
 * CONSTANTS[GNaL_b] is GNaL_b in component INaL (milliS_per_microF).
 * CONSTANTS[GNaL] is GNaL in component INaL (milliS_per_microF).
 * ALGEBRAIC[fINaLp] is fINaLp in component INaL (dimensionless).
 * CONSTANTS[Gto_b] is Gto_b in component Ito (milliS_per_microF).
 * ALGEBRAIC[ass] is ass in component Ito (dimensionless).
 * ALGEBRAIC[ta] is ta in component Ito (millisecond).
 * STATES[a] is a in component Ito (dimensionless).
 * ALGEBRAIC[iss] is iss in component Ito (dimensionless).
 * ALGEBRAIC[delta_epi] is delta_epi in component Ito (dimensionless).
 * ALGEBRAIC[tiF_b] is tiF_b in component Ito (millisecond).
 * ALGEBRAIC[tiS_b] is tiS_b in component Ito (millisecond).
 * ALGEBRAIC[tiF] is tiF in component Ito (millisecond).
 * ALGEBRAIC[tiS] is tiS in component Ito (millisecond).
 * ALGEBRAIC[AiF] is AiF in component Ito (dimensionless).
 * ALGEBRAIC[AiS] is AiS in component Ito (dimensionless).
 * STATES[iF] is iF in component Ito (dimensionless).
 * STATES[iS] is iS in component Ito (dimensionless).
 * ALGEBRAIC[i] is i in component Ito (dimensionless).
 * ALGEBRAIC[assp] is assp in component Ito (dimensionless).
 * STATES[ap] is ap in component Ito (dimensionless).
 * ALGEBRAIC[dti_develop] is dti_develop in component Ito (dimensionless).
 * ALGEBRAIC[dti_recover] is dti_recover in component Ito (dimensionless).
 * ALGEBRAIC[tiFp] is tiFp in component Ito (millisecond).
 * ALGEBRAIC[tiSp] is tiSp in component Ito (millisecond).
 * STATES[iFp] is iFp in component Ito (dimensionless).
 * STATES[iSp] is iSp in component Ito (dimensionless).
 * ALGEBRAIC[ip] is ip in component Ito (dimensionless).
 * CONSTANTS[Gto] is Gto in component Ito (milliS_per_microF).
 * ALGEBRAIC[fItop] is fItop in component Ito (dimensionless).
 * CONSTANTS[Kmn] is Kmn in component ICaL (millimolar).
 * CONSTANTS[k2n] is k2n in component ICaL (per_millisecond).
 * CONSTANTS[PCa_b] is PCa_b in component ICaL (dimensionless).
 * ALGEBRAIC[dss] is dss in component ICaL (dimensionless).
 * STATES[d] is d in component ICaL (dimensionless).
 * ALGEBRAIC[fss] is fss in component ICaL (dimensionless).
 * CONSTANTS[Aff] is Aff in component ICaL (dimensionless).
 * CONSTANTS[Afs] is Afs in component ICaL (dimensionless).
 * STATES[ff] is ff in component ICaL (dimensionless).
 * STATES[fs] is fs in component ICaL (dimensionless).
 * ALGEBRAIC[f] is f in component ICaL (dimensionless).
 * ALGEBRAIC[fcass] is fcass in component ICaL (dimensionless).
 * ALGEBRAIC[Afcaf] is Afcaf in component ICaL (dimensionless).
 * ALGEBRAIC[Afcas] is Afcas in component ICaL (dimensionless).
 * STATES[fcaf] is fcaf in component ICaL (dimensionless).
 * STATES[fcas] is fcas in component ICaL (dimensionless).
 * ALGEBRAIC[fca] is fca in component ICaL (dimensionless).
 * STATES[jca] is jca in component ICaL (dimensionless).
 * STATES[ffp] is ffp in component ICaL (dimensionless).
 * ALGEBRAIC[fp] is fp in component ICaL (dimensionless).
 * STATES[fcafp] is fcafp in component ICaL (dimensionless).
 * ALGEBRAIC[fcap] is fcap in component ICaL (dimensionless).
 * ALGEBRAIC[km2n] is km2n in component ICaL (per_millisecond).
 * ALGEBRAIC[anca] is anca in component ICaL (dimensionless).
 * STATES[nca] is nca in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaL] is PhiCaL in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaNa] is PhiCaNa in component ICaL (dimensionless).
 * ALGEBRAIC[PhiCaK] is PhiCaK in component ICaL (dimensionless).
 * CONSTANTS[PCa] is PCa in component ICaL (dimensionless).
 * CONSTANTS[PCap] is PCap in component ICaL (dimensionless).
 * CONSTANTS[PCaNa] is PCaNa in component ICaL (dimensionless).
 * CONSTANTS[PCaK] is PCaK in component ICaL (dimensionless).
 * CONSTANTS[PCaNap] is PCaNap in component ICaL (dimensionless).
 * CONSTANTS[PCaKp] is PCaKp in component ICaL (dimensionless).
 * ALGEBRAIC[fICaLp] is fICaLp in component ICaL (dimensionless).
 * ALGEBRAIC[td] is td in component ICaL (millisecond).
 * ALGEBRAIC[tff] is tff in component ICaL (millisecond).
 * ALGEBRAIC[tfs] is tfs in component ICaL (millisecond).
 * ALGEBRAIC[tfcaf] is tfcaf in component ICaL (millisecond).
 * ALGEBRAIC[tfcas] is tfcas in component ICaL (millisecond).
 * CONSTANTS[tjca] is tjca in component ICaL (millisecond).
 * ALGEBRAIC[tffp] is tffp in component ICaL (millisecond).
 * ALGEBRAIC[tfcafp] is tfcafp in component ICaL (millisecond).
 * CONSTANTS[v0_CaL] is v0 in component ICaL (millivolt).
 * ALGEBRAIC[A_1] is A_1 in component ICaL (dimensionless).
 * CONSTANTS[B_1] is B_1 in component ICaL (per_millivolt).
 * ALGEBRAIC[U_1] is U_1 in component ICaL (dimensionless).
 * ALGEBRAIC[A_2] is A_2 in component ICaL (dimensionless).
 * CONSTANTS[B_2] is B_2 in component ICaL (per_millivolt).
 * ALGEBRAIC[U_2] is U_2 in component ICaL (dimensionless).
 * ALGEBRAIC[A_3] is A_3 in component ICaL (dimensionless).
 * CONSTANTS[B_3] is B_3 in component ICaL (per_millivolt).
 * ALGEBRAIC[U_3] is U_3 in component ICaL (dimensionless).
 * CONSTANTS[GKr_b] is GKr_b in component IKr (milliS_per_microF).
 * STATES[IC1] is IC1 in component IKr (dimensionless).
 * STATES[IC2] is IC2 in component IKr (dimensionless).
 * STATES[C1] is C1 in component IKr (dimensionless).
 * STATES[C2] is C2 in component IKr (dimensionless).
 * STATES[O] is O in component IKr (dimensionless).
 * STATES[IO] is IO in component IKr (dimensionless).
 * STATES[IObound] is IObound in component IKr (dimensionless).
 * STATES[Obound] is Obound in component IKr (dimensionless).
 * STATES[Cbound] is Cbound in component IKr (dimensionless).
 * STATES[D] is D in component IKr (dimensionless).
 * CONSTANTS[GKr] is GKr in component IKr (milliS_per_microF).
 * CONSTANTS[A1] is A1 in component IKr (per_millisecond).
 * CONSTANTS[B1] is B1 in component IKr (per_millivolt).
 * CONSTANTS[q1] is q1 in component IKr (dimensionless).
 * CONSTANTS[A2] is A2 in component IKr (per_millisecond).
 * CONSTANTS[B2] is B2 in component IKr (per_millivolt).
 * CONSTANTS[q2] is q2 in component IKr (dimensionless).
 * CONSTANTS[A3] is A3 in component IKr (per_millisecond).
 * CONSTANTS[B3] is B3 in component IKr (per_millivolt).
 * CONSTANTS[q3] is q3 in component IKr (dimensionless).
 * CONSTANTS[A4] is A4 in component IKr (per_millisecond).
 * CONSTANTS[B4] is B4 in component IKr (per_millivolt).
 * CONSTANTS[q4] is q4 in component IKr (dimensionless).
 * CONSTANTS[A11] is A11 in component IKr (per_millisecond).
 * CONSTANTS[B11] is B11 in component IKr (per_millivolt).
 * CONSTANTS[q11] is q11 in component IKr (dimensionless).
 * CONSTANTS[A21] is A21 in component IKr (per_millisecond).
 * CONSTANTS[B21] is B21 in component IKr (per_millivolt).
 * CONSTANTS[q21] is q21 in component IKr (dimensionless).
 * CONSTANTS[A31] is A31 in component IKr (per_millisecond).
 * CONSTANTS[B31] is B31 in component IKr (per_millivolt).
 * CONSTANTS[q31] is q31 in component IKr (dimensionless).
 * CONSTANTS[A41] is A41 in component IKr (per_millisecond).
 * CONSTANTS[B41] is B41 in component IKr (per_millivolt).
 * CONSTANTS[q41] is q41 in component IKr (dimensionless).
 * CONSTANTS[A51] is A51 in component IKr (per_millisecond).
 * CONSTANTS[B51] is B51 in component IKr (per_millivolt).
 * CONSTANTS[q51] is q51 in component IKr (dimensionless).
 * CONSTANTS[A52] is A52 in component IKr (per_millisecond).
 * CONSTANTS[B52] is B52 in component IKr (per_millivolt).
 * CONSTANTS[q52] is q52 in component IKr (dimensionless).
 * CONSTANTS[A53] is A53 in component IKr (per_millisecond).
 * CONSTANTS[B53] is B53 in component IKr (per_millivolt).
 * CONSTANTS[q53] is q53 in component IKr (dimensionless).
 * CONSTANTS[A61] is A61 in component IKr (per_millisecond).
 * CONSTANTS[B61] is B61 in component IKr (per_millivolt).
 * CONSTANTS[q61] is q61 in component IKr (dimensionless).
 * CONSTANTS[A62] is A62 in component IKr (per_millisecond).
 * CONSTANTS[B62] is B62 in component IKr (per_millivolt).
 * CONSTANTS[q62] is q62 in component IKr (dimensionless).
 * CONSTANTS[A63] is A63 in component IKr (per_millisecond).
 * CONSTANTS[B63] is B63 in component IKr (per_millivolt).
 * CONSTANTS[q63] is q63 in component IKr (dimensionless).
 * CONSTANTS[Kmax] is Kmax in component IKr (dimensionless).
 * CONSTANTS[Ku] is Ku in component IKr (per_millisecond).
 * CONSTANTS[n] is n in component IKr (dimensionless).
 * CONSTANTS[halfmax] is halfmax in component IKr (dimensionless).
 * CONSTANTS[Kt] is Kt in component IKr (per_millisecond).
 * CONSTANTS[Vhalf] is Vhalf in component IKr (millivolt).
 * CONSTANTS[Temp] is Temp in component IKr (dimensionless).
 * CONSTANTS[GKs_b] is GKs_b in component IKs (milliS_per_microF).
 * CONSTANTS[GKs] is GKs in component IKs (milliS_per_microF).
 * ALGEBRAIC[xs1ss] is xs1ss in component IKs (dimensionless).
 * ALGEBRAIC[xs2ss] is xs2ss in component IKs (dimensionless).
 * ALGEBRAIC[txs1] is txs1 in component IKs (millisecond).
 * CONSTANTS[txs1_max] is txs1_max in component IKs (millisecond).
 * STATES[xs1] is xs1 in component IKs (dimensionless).
 * STATES[xs2] is xs2 in component IKs (dimensionless).
 * ALGEBRAIC[KsCa] is KsCa in component IKs (dimensionless).
 * ALGEBRAIC[txs2] is txs2 in component IKs (millisecond).
 * CONSTANTS[GK1] is GK1 in component IK1 (milliS_per_microF).
 * CONSTANTS[GK1_b] is GK1_b in component IK1 (milliS_per_microF).
 * ALGEBRAIC[xk1ss] is xk1ss in component IK1 (dimensionless).
 * ALGEBRAIC[txk1] is txk1 in component IK1 (millisecond).
 * STATES[xk1] is xk1 in component IK1 (dimensionless).
 * ALGEBRAIC[rk1] is rk1 in component IK1 (millisecond).
 * CONSTANTS[kna1] is kna1 in component INaCa_i (per_millisecond).
 * CONSTANTS[kna2] is kna2 in component INaCa_i (per_millisecond).
 * CONSTANTS[kna3] is kna3 in component INaCa_i (per_millisecond).
 * CONSTANTS[kasymm] is kasymm in component INaCa_i (dimensionless).
 * CONSTANTS[wna] is wna in component INaCa_i (dimensionless).
 * CONSTANTS[wca] is wca in component INaCa_i (dimensionless).
 * CONSTANTS[wnaca] is wnaca in component INaCa_i (dimensionless).
 * CONSTANTS[kcaon] is kcaon in component INaCa_i (per_millisecond).
 * CONSTANTS[kcaoff] is kcaoff in component INaCa_i (per_millisecond).
 * CONSTANTS[qna] is qna in component INaCa_i (dimensionless).
 * CONSTANTS[qca] is qca in component INaCa_i (dimensionless).
 * ALGEBRAIC[hna] is hna in component INaCa_i (dimensionless).
 * ALGEBRAIC[hca] is hca in component INaCa_i (dimensionless).
 * CONSTANTS[KmCaAct] is KmCaAct in component INaCa_i (millimolar).
 * CONSTANTS[Gncx_b] is Gncx_b in component INaCa_i (milliS_per_microF).
 * CONSTANTS[Gncx] is Gncx in component INaCa_i (milliS_per_microF).
 * ALGEBRAIC[h1_i] is h1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h2_i] is h2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h3_i] is h3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h4_i] is h4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h5_i] is h5_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h6_i] is h6_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h7_i] is h7_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h8_i] is h8_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[h9_i] is h9_i in component INaCa_i (dimensionless).
 * CONSTANTS[h10_i] is h10_i in component INaCa_i (dimensionless).
 * CONSTANTS[h11_i] is h11_i in component INaCa_i (dimensionless).
 * CONSTANTS[h12_i] is h12_i in component INaCa_i (dimensionless).
 * CONSTANTS[k1_i] is k1_i in component INaCa_i (dimensionless).
 * CONSTANTS[k2_i] is k2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3p_i] is k3p_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3pp_i] is k3pp_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3_i] is k3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4_i] is k4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4p_i] is k4p_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4pp_i] is k4pp_i in component INaCa_i (dimensionless).
 * CONSTANTS[k5_i] is k5_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k6_i] is k6_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k7_i] is k7_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[k8_i] is k8_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x1_i] is x1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x2_i] is x2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x3_i] is x3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[x4_i] is x4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E1_i] is E1_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E2_i] is E2_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E3_i] is E3_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[E4_i] is E4_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[allo_i] is allo_i in component INaCa_i (dimensionless).
 * ALGEBRAIC[JncxNa_i] is JncxNa_i in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_i] is JncxCa_i in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[h1_ss] is h1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h2_ss] is h2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h3_ss] is h3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h4_ss] is h4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h5_ss] is h5_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h6_ss] is h6_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h7_ss] is h7_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h8_ss] is h8_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[h9_ss] is h9_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h10_ss] is h10_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h11_ss] is h11_ss in component INaCa_i (dimensionless).
 * CONSTANTS[h12_ss] is h12_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k1_ss] is k1_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k2_ss] is k2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3p_ss] is k3p_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3pp_ss] is k3pp_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k3_ss] is k3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4_ss] is k4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4p_ss] is k4p_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k4pp_ss] is k4pp_ss in component INaCa_i (dimensionless).
 * CONSTANTS[k5_ss] is k5_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k6_ss] is k6_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k7_ss] is k7_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[k8_ss] is k8_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x1_ss] is x1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x2_ss] is x2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x3_ss] is x3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[x4_ss] is x4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E1_ss] is E1_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E2_ss] is E2_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E3_ss] is E3_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[E4_ss] is E4_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[allo_ss] is allo_ss in component INaCa_i (dimensionless).
 * ALGEBRAIC[JncxNa_ss] is JncxNa_ss in component INaCa_i (millimolar_per_millisecond).
 * ALGEBRAIC[JncxCa_ss] is JncxCa_ss in component INaCa_i (millimolar_per_millisecond).
 * CONSTANTS[k1p] is k1p in component INaK (per_millisecond).
 * CONSTANTS[k1m] is k1m in component INaK (per_millisecond).
 * CONSTANTS[k2p] is k2p in component INaK (per_millisecond).
 * CONSTANTS[k2m] is k2m in component INaK (per_millisecond).
 * CONSTANTS[k3p] is k3p in component INaK (per_millisecond).
 * CONSTANTS[k3m] is k3m in component INaK (per_millisecond).
 * CONSTANTS[k4p] is k4p in component INaK (per_millisecond).
 * CONSTANTS[k4m] is k4m in component INaK (per_millisecond).
 * CONSTANTS[Knai0] is Knai0 in component INaK (millimolar).
 * CONSTANTS[Knao0] is Knao0 in component INaK (millimolar).
 * CONSTANTS[delta] is delta in component INaK (millivolt).
 * CONSTANTS[Kki] is Kki in component INaK (per_millisecond).
 * CONSTANTS[Kko] is Kko in component INaK (per_millisecond).
 * CONSTANTS[MgADP] is MgADP in component INaK (millimolar).
 * CONSTANTS[MgATP] is MgATP in component INaK (millimolar).
 * CONSTANTS[Kmgatp] is Kmgatp in component INaK (millimolar).
 * CONSTANTS[H] is H in component INaK (millimolar).
 * CONSTANTS[eP] is eP in component INaK (dimensionless).
 * CONSTANTS[Khp] is Khp in component INaK (millimolar).
 * CONSTANTS[Knap] is Knap in component INaK (millimolar).
 * CONSTANTS[Kxkur] is Kxkur in component INaK (millimolar).
 * CONSTANTS[Pnak_b] is Pnak_b in component INaK (milliS_per_microF).
 * CONSTANTS[Pnak] is Pnak in component INaK (milliS_per_microF).
 * ALGEBRAIC[Knai] is Knai in component INaK (millimolar).
 * ALGEBRAIC[Knao] is Knao in component INaK (millimolar).
 * ALGEBRAIC[P] is P in component INaK (dimensionless).
 * ALGEBRAIC[a1] is a1 in component INaK (dimensionless).
 * CONSTANTS[b1] is b1 in component INaK (dimensionless).
 * CONSTANTS[a2] is a2 in component INaK (dimensionless).
 * ALGEBRAIC[b2] is b2 in component INaK (dimensionless).
 * ALGEBRAIC[a3] is a3 in component INaK (dimensionless).
 * ALGEBRAIC[b3] is b3 in component INaK (dimensionless).
 * CONSTANTS[a4] is a4 in component INaK (dimensionless).
 * ALGEBRAIC[b4] is b4 in component INaK (dimensionless).
 * ALGEBRAIC[x1] is x1 in component INaK (dimensionless).
 * ALGEBRAIC[x2] is x2 in component INaK (dimensionless).
 * ALGEBRAIC[x3] is x3 in component INaK (dimensionless).
 * ALGEBRAIC[x4] is x4 in component INaK (dimensionless).
 * ALGEBRAIC[E1] is E1 in component INaK (dimensionless).
 * ALGEBRAIC[E2] is E2 in component INaK (dimensionless).
 * ALGEBRAIC[E3] is E3 in component INaK (dimensionless).
 * ALGEBRAIC[E4] is E4 in component INaK (dimensionless).
 * ALGEBRAIC[JnakNa] is JnakNa in component INaK (millimolar_per_millisecond).
 * ALGEBRAIC[JnakK] is JnakK in component INaK (millimolar_per_millisecond).
 * ALGEBRAIC[xkb] is xkb in component IKb (dimensionless).
 * CONSTANTS[GKb_b] is GKb_b in component IKb (milliS_per_microF).
 * CONSTANTS[GKb] is GKb in component IKb (milliS_per_microF).
 * CONSTANTS[PNab] is PNab in component INab (milliS_per_microF).
 * ALGEBRAIC[A_Nab] is A in component INab (microA_per_microF).
 * CONSTANTS[B_Nab] is B in component INab (per_millivolt).
 * CONSTANTS[v0_Nab] is v0 in component INab (millivolt).
 * ALGEBRAIC[U] is U in component INab (dimensionless).
 * CONSTANTS[PCab] is PCab in component ICab (milliS_per_microF).
 * ALGEBRAIC[A_Cab] is A in component ICab (microA_per_microF).
 * CONSTANTS[B_Cab] is B in component ICab (per_millivolt).
 * CONSTANTS[v0_Cab] is v0 in component ICab (millivolt).
 * ALGEBRAIC[U] is U in component ICab (dimensionless).
 * CONSTANTS[GpCa] is GpCa in component IpCa (milliS_per_microF).
 * CONSTANTS[KmCap] is KmCap in component IpCa (millimolar).
 * CONSTANTS[bt] is bt in component ryr (millisecond).
 * CONSTANTS[a_rel] is a_rel in component ryr (millisecond).
 * ALGEBRAIC[Jrel_inf] is Jrel_inf in component ryr (dimensionless).
 * ALGEBRAIC[tau_rel] is tau_rel in component ryr (millisecond).
 * ALGEBRAIC[Jrel_infp] is Jrel_infp in component ryr (dimensionless).
 * ALGEBRAIC[Jrel_temp] is Jrel_temp in component ryr (dimensionless).
 * ALGEBRAIC[tau_relp] is tau_relp in component ryr (millisecond).
 * STATES[Jrelnp] is Jrelnp in component ryr (dimensionless).
 * STATES[Jrelp] is Jrelp in component ryr (dimensionless).
 * CONSTANTS[btp] is btp in component ryr (millisecond).
 * CONSTANTS[a_relp] is a_relp in component ryr (millisecond).
 * ALGEBRAIC[Jrel_inf_temp] is Jrel_inf_temp in component ryr (dimensionless).
 * ALGEBRAIC[fJrelp] is fJrelp in component ryr (dimensionless).
 * CONSTANTS[Jrel_scaling_factor] is Jrel_scaling_factor in component ryr (dimensionless).
 * ALGEBRAIC[tau_rel_temp] is tau_rel_temp in component ryr (millisecond).
 * ALGEBRAIC[tau_relp_temp] is tau_relp_temp in component ryr (millisecond).
 * CONSTANTS[upScale] is upScale in component SERCA (dimensionless).
 * ALGEBRAIC[Jupnp] is Jupnp in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[Jupp] is Jupp in component SERCA (millimolar_per_millisecond).
 * ALGEBRAIC[fJupp] is fJupp in component SERCA (dimensionless).
 * ALGEBRAIC[Jleak] is Jleak in component SERCA (millimolar_per_millisecond).
 * CONSTANTS[Jup_b] is Jup_b in component SERCA (dimensionless).
 * RATES[V] is d/dt v in component membrane (millivolt).
 * RATES[CaMKt] is d/dt CaMKt in component CaMK (millimolar).
 * RATES[nai] is d/dt nai in component intracellular_ions (millimolar).
 * RATES[nass] is d/dt nass in component intracellular_ions (millimolar).
 * RATES[ki] is d/dt ki in component intracellular_ions (millimolar).
 * RATES[kss] is d/dt kss in component intracellular_ions (millimolar).
 * RATES[cai] is d/dt cai in component intracellular_ions (millimolar).
 * RATES[cass] is d/dt cass in component intracellular_ions (millimolar).
 * RATES[cansr] is d/dt cansr in component intracellular_ions (millimolar).
 * RATES[cajsr] is d/dt cajsr in component intracellular_ions (millimolar).
 * RATES[m] is d/dt m in component INa (dimensionless).
 * RATES[hf] is d/dt hf in component INa (dimensionless).
 * RATES[hs] is d/dt hs in component INa (dimensionless).
 * RATES[j] is d/dt j in component INa (dimensionless).
 * RATES[hsp] is d/dt hsp in component INa (dimensionless).
 * RATES[jp] is d/dt jp in component INa (dimensionless).
 * RATES[mL] is d/dt mL in component INaL (dimensionless).
 * RATES[hL] is d/dt hL in component INaL (dimensionless).
 * RATES[hLp] is d/dt hLp in component INaL (dimensionless).
 * RATES[a] is d/dt a in component Ito (dimensionless).
 * RATES[iF] is d/dt iF in component Ito (dimensionless).
 * RATES[iS] is d/dt iS in component Ito (dimensionless).
 * RATES[ap] is d/dt ap in component Ito (dimensionless).
 * RATES[iFp] is d/dt iFp in component Ito (dimensionless).
 * RATES[iSp] is d/dt iSp in component Ito (dimensionless).
 * RATES[d] is d/dt d in component ICaL (dimensionless).
 * RATES[ff] is d/dt ff in component ICaL (dimensionless).
 * RATES[fs] is d/dt fs in component ICaL (dimensionless).
 * RATES[fcaf] is d/dt fcaf in component ICaL (dimensionless).
 * RATES[fcas] is d/dt fcas in component ICaL (dimensionless).
 * RATES[jca] is d/dt jca in component ICaL (dimensionless).
 * RATES[ffp] is d/dt ffp in component ICaL (dimensionless).
 * RATES[fcafp] is d/dt fcafp in component ICaL (dimensionless).
 * RATES[nca] is d/dt nca in component ICaL (dimensionless).
 * RATES[IC1] is d/dt IC1 in component IKr (dimensionless).
 * RATES[IC2] is d/dt IC2 in component IKr (dimensionless).
 * RATES[C1] is d/dt C1 in component IKr (dimensionless).
 * RATES[C2] is d/dt C2 in component IKr (dimensionless).
 * RATES[O] is d/dt O in component IKr (dimensionless).
 * RATES[IO] is d/dt IO in component IKr (dimensionless).
 * RATES[IObound] is d/dt IObound in component IKr (dimensionless).
 * RATES[Obound] is d/dt Obound in component IKr (dimensionless).
 * RATES[Cbound] is d/dt Cbound in component IKr (dimensionless).
 * RATES[D] is d/dt D in component IKr (dimensionless).
 * RATES[xs1] is d/dt xs1 in component IKs (dimensionless).
 * RATES[xs2] is d/dt xs2 in component IKs (dimensionless).
 * RATES[xk1] is d/dt xk1 in component IK1 (dimensionless).
 * RATES[Jrelnp] is d/dt Jrelnp in component ryr (dimensionless).
 * RATES[Jrelp] is d/dt Jrelp in component ryr (dimensionless).
 */


  // algebraic_size = 200;
  // constants_size= 208;
  // states_size = 49;

__device__ void ___initConsts(double *CONSTANTS, double *STATES, double type, double bcl, int offset)
{
CONSTANTS[(208 * offset) +  celltype] = type;
CONSTANTS[(208 * offset) +  nao] = 140;
CONSTANTS[(208 * offset) +  cao] = 1.8;
CONSTANTS[(208 * offset) +  ko] = 5.4;
CONSTANTS[(208 * offset) +  R] = 8314;
CONSTANTS[(208 * offset) +  T] = 310;
CONSTANTS[(208 * offset) +  F] = 96485;
CONSTANTS[(208 * offset) +  zna] = 1;
CONSTANTS[(208 * offset) +  zca] = 2;
CONSTANTS[(208 * offset) +  zk] = 1;
CONSTANTS[(208 * offset) +  L] = 0.01;
CONSTANTS[(208 * offset) +  rad] = 0.0011;
STATES[(49 * offset) + V] = -88.00190;
CONSTANTS[(208 * offset) +  stim_start] = 10;
CONSTANTS[(208 * offset) +  stim_end] = 100000000000000000;
CONSTANTS[(208 * offset) +  amp] = -80;
CONSTANTS[(208 * offset) +  BCL] = 1000;
CONSTANTS[(208 * offset) +  duration] = 0.5;
CONSTANTS[(208 * offset) +  KmCaMK] = 0.15;
CONSTANTS[(208 * offset) +  aCaMK] = 0.05;
CONSTANTS[(208 * offset) +  bCaMK] = 0.00068;
CONSTANTS[(208 * offset) +  CaMKo] = 0.05;
CONSTANTS[(208 * offset) +  KmCaM] = 0.0015;
STATES[(49 * offset) + CaMKt] = 0.0125840447;
STATES[(49 * offset) + cass] = 8.49e-05;
CONSTANTS[(208 * offset) +  cmdnmax_b] = 0.05;
CONSTANTS[(208 * offset) +  kmcmdn] = 0.00238;
CONSTANTS[(208 * offset) +  trpnmax] = 0.07;
CONSTANTS[(208 * offset) +  kmtrpn] = 0.0005;
CONSTANTS[(208 * offset) +  BSRmax] = 0.047;
CONSTANTS[(208 * offset) +  KmBSR] = 0.00087;
CONSTANTS[(208 * offset) +  BSLmax] = 1.124;
CONSTANTS[(208 * offset) +  KmBSL] = 0.0087;
CONSTANTS[(208 * offset) +  csqnmax] = 10;
CONSTANTS[(208 * offset) +  kmcsqn] = 0.8;
STATES[(49 * offset) + nai] = 7.268004498;
STATES[(49 * offset) + nass] = 7.268089977;
STATES[(49 * offset) + ki] = 144.6555918;
STATES[(49 * offset) + kss] = 144.6555651;
STATES[(49 * offset) + cansr] = 1.619574538;
STATES[(49 * offset) + cajsr] = 1.571234014;
STATES[(49 * offset) + cai] = 8.6e-05;
CONSTANTS[(208 * offset) +  cm] = 1;
CONSTANTS[(208 * offset) +  PKNa] = 0.01833;
CONSTANTS[(208 * offset) +  mssV1] = 39.57;
CONSTANTS[(208 * offset) +  mssV2] = 9.871;
CONSTANTS[(208 * offset) +  mtV1] = 11.64;
CONSTANTS[(208 * offset) +  mtV2] = 34.77;
CONSTANTS[(208 * offset) +  mtD1] = 6.765;
CONSTANTS[(208 * offset) +  mtD2] = 8.552;
CONSTANTS[(208 * offset) +  mtV3] = 77.42;
CONSTANTS[(208 * offset) +  mtV4] = 5.955;
STATES[(49 * offset) + m] = 0.007344121102;
CONSTANTS[(208 * offset) +  hssV1] = 82.9;
CONSTANTS[(208 * offset) +  hssV2] = 6.086;
CONSTANTS[(208 * offset) +  Ahf] = 0.99;
STATES[(49 * offset) + hf] = 0.6981071913;
STATES[(49 * offset) + hs] = 0.6980895801;
CONSTANTS[(208 * offset) +  GNa] = 75;
CONSTANTS[(208 * offset) +  shift_INa_inact] = 0;
STATES[(49 * offset) + j] = 0.6979908432;
STATES[(49 * offset) + hsp] = 0.4549485525;
STATES[(49 * offset) + jp] = 0.6979245865;
STATES[(49 * offset) + mL] = 0.0001882617273;
CONSTANTS[(208 * offset) +  thL] = 200;
STATES[(49 * offset) + hL] = 0.5008548855;
STATES[(49 * offset) + hLp] = 0.2693065357;
CONSTANTS[(208 * offset) +  GNaL_b] = 0.019957499999999975;
CONSTANTS[(208 * offset) +  Gto_b] = 0.02;
STATES[(49 * offset) + a] = 0.001001097687;
STATES[(49 * offset) + iF] = 0.9995541745;
STATES[(49 * offset) + iS] = 0.5865061736;
STATES[(49 * offset) + ap] = 0.0005100862934;
STATES[(49 * offset) + iFp] = 0.9995541823;
STATES[(49 * offset) + iSp] = 0.6393399482;
CONSTANTS[(208 * offset) +  Kmn] = 0.002;
CONSTANTS[(208 * offset) +  k2n] = 1000;
CONSTANTS[(208 * offset) +  PCa_b] = 0.0001007;
STATES[(49 * offset) + d] = 2.34e-9;
STATES[(49 * offset) + ff] = 0.9999999909;
STATES[(49 * offset) + fs] = 0.9102412777;
STATES[(49 * offset) + fcaf] = 0.9999999909;
STATES[(49 * offset) + fcas] = 0.9998046777;
STATES[(49 * offset) + jca] = 0.9999738312;
STATES[(49 * offset) + ffp] = 0.9999999909;
STATES[(49 * offset) + fcafp] = 0.9999999909;
STATES[(49 * offset) + nca] = 0.002749414044;
CONSTANTS[(208 * offset) +  GKr_b] = 0.04658545454545456;
STATES[(49 * offset) + IC1] = 0.999637;
STATES[(49 * offset) + IC2] = 6.83208e-05;
STATES[(49 * offset) + C1] = 1.80145e-08;
STATES[(49 * offset) + C2] = 8.26619e-05;
STATES[(49 * offset) + O] = 0.00015551;
STATES[(49 * offset) + IO] = 5.67623e-05;
STATES[(49 * offset) + IObound] = 0;
STATES[(49 * offset) + Obound] = 0;
STATES[(49 * offset) + Cbound] = 0;
STATES[(49 * offset) + D] = 0;
CONSTANTS[(208 * offset) +  A1] = 0.0264;
CONSTANTS[(208 * offset) +  B1] = 4.631E-05;
CONSTANTS[(208 * offset) +  q1] = 4.843;
CONSTANTS[(208 * offset) +  A2] = 4.986E-06;
CONSTANTS[(208 * offset) +  B2] = -0.004226;
CONSTANTS[(208 * offset) +  q2] = 4.23;
CONSTANTS[(208 * offset) +  A3] = 0.001214;
CONSTANTS[(208 * offset) +  B3] = 0.008516;
CONSTANTS[(208 * offset) +  q3] = 4.962;
CONSTANTS[(208 * offset) +  A4] = 1.854E-05;
CONSTANTS[(208 * offset) +  B4] = -0.04641;
CONSTANTS[(208 * offset) +  q4] = 3.769;
CONSTANTS[(208 * offset) +  A11] = 0.0007868;
CONSTANTS[(208 * offset) +  B11] = 1.535E-08;
CONSTANTS[(208 * offset) +  q11] = 4.942;
CONSTANTS[(208 * offset) +  A21] = 5.455E-06;
CONSTANTS[(208 * offset) +  B21] = -0.1688;
CONSTANTS[(208 * offset) +  q21] = 4.156;
CONSTANTS[(208 * offset) +  A31] = 0.005509;
CONSTANTS[(208 * offset) +  B31] = 7.771E-09;
CONSTANTS[(208 * offset) +  q31] = 4.22;
CONSTANTS[(208 * offset) +  A41] = 0.001416;
CONSTANTS[(208 * offset) +  B41] = -0.02877;
CONSTANTS[(208 * offset) +  q41] = 1.459;
CONSTANTS[(208 * offset) +  A51] = 0.4492;
CONSTANTS[(208 * offset) +  B51] = 0.008595;
CONSTANTS[(208 * offset) +  q51] = 5;
CONSTANTS[(208 * offset) +  A52] = 0.3181;
CONSTANTS[(208 * offset) +  B52] = 3.613E-08;
CONSTANTS[(208 * offset) +  q52] = 4.663;
CONSTANTS[(208 * offset) +  A53] = 0.149;
CONSTANTS[(208 * offset) +  B53] = 0.004668;
CONSTANTS[(208 * offset) +  q53] = 2.412;
CONSTANTS[(208 * offset) +  A61] = 0.01241;
CONSTANTS[(208 * offset) +  B61] = 0.1725;
CONSTANTS[(208 * offset) +  q61] = 5.568;
CONSTANTS[(208 * offset) +  A62] = 0.3226;
CONSTANTS[(208 * offset) +  B62] = -0.0006575;
CONSTANTS[(208 * offset) +  q62] = 5;
CONSTANTS[(208 * offset) +  A63] = 0.008978;
CONSTANTS[(208 * offset) +  B63] = -0.02215;
CONSTANTS[(208 * offset) +  q63] = 5.682;
CONSTANTS[(208 * offset) +  Kmax] = 0;
CONSTANTS[(208 * offset) +  Ku] = 0;
CONSTANTS[(208 * offset) +  n] = 1;
CONSTANTS[(208 * offset) +  halfmax] = 1;
CONSTANTS[(208 * offset) +  Kt] = 0;
CONSTANTS[(208 * offset) +  Vhalf] = 1;
CONSTANTS[(208 * offset) +  Temp] = 37;
CONSTANTS[(208 * offset) +  GKs_b] = 0.006358000000000001;
CONSTANTS[(208 * offset) +  txs1_max] = 817.3;
STATES[(49 * offset) + xs1] = 0.2707758025;
STATES[(49 * offset) + xs2] = 0.0001928503426;
CONSTANTS[(208 * offset) +  GK1_b] = 0.3239783999999998;
STATES[(49 * offset) + xk1] = 0.9967597594;
CONSTANTS[(208 * offset) +  kna1] = 15;
CONSTANTS[(208 * offset) +  kna2] = 5;
CONSTANTS[(208 * offset) +  kna3] = 88.12;
CONSTANTS[(208 * offset) +  kasymm] = 12.5;
CONSTANTS[(208 * offset) +  wna] = 6e4;
CONSTANTS[(208 * offset) +  wca] = 6e4;
CONSTANTS[(208 * offset) +  wnaca] = 5e3;
CONSTANTS[(208 * offset) +  kcaon] = 1.5e6;
CONSTANTS[(208 * offset) +  kcaoff] = 5e3;
CONSTANTS[(208 * offset) +  qna] = 0.5224;
CONSTANTS[(208 * offset) +  qca] = 0.167;
CONSTANTS[(208 * offset) +  KmCaAct] = 150e-6;
CONSTANTS[(208 * offset) +  Gncx_b] = 0.0008;
CONSTANTS[(208 * offset) +  k1p] = 949.5;
CONSTANTS[(208 * offset) +  k1m] = 182.4;
CONSTANTS[(208 * offset) +  k2p] = 687.2;
CONSTANTS[(208 * offset) +  k2m] = 39.4;
CONSTANTS[(208 * offset) +  k3p] = 1899;
CONSTANTS[(208 * offset) +  k3m] = 79300;
CONSTANTS[(208 * offset) +  k4p] = 639;
CONSTANTS[(208 * offset) +  k4m] = 40;
CONSTANTS[(208 * offset) +  Knai0] = 9.073;
CONSTANTS[(208 * offset) +  Knao0] = 27.78;
CONSTANTS[(208 * offset) +  delta] = -0.155;
CONSTANTS[(208 * offset) +  Kki] = 0.5;
CONSTANTS[(208 * offset) +  Kko] = 0.3582;
CONSTANTS[(208 * offset) +  MgADP] = 0.05;
CONSTANTS[(208 * offset) +  MgATP] = 9.8;
CONSTANTS[(208 * offset) +  Kmgatp] = 1.698e-7;
CONSTANTS[(208 * offset) +  H] = 1e-7;
CONSTANTS[(208 * offset) +  eP] = 4.2;
CONSTANTS[(208 * offset) +  Khp] = 1.698e-7;
CONSTANTS[(208 * offset) +  Knap] = 224;
CONSTANTS[(208 * offset) +  Kxkur] = 292;
CONSTANTS[(208 * offset) +  Pnak_b] = 30;
CONSTANTS[(208 * offset) +  GKb_b] = 0.003;
CONSTANTS[(208 * offset) +  PNab] = 3.75e-10;
CONSTANTS[(208 * offset) +  PCab] = 2.5e-8;
CONSTANTS[(208 * offset) +  GpCa] = 0.0005;
CONSTANTS[(208 * offset) +  KmCap] = 0.0005;
CONSTANTS[(208 * offset) +  bt] = 4.75;
STATES[(49 * offset) + Jrelnp] = 2.5e-7;
STATES[(49 * offset) + Jrelp] = 3.12e-7;
CONSTANTS[(208 * offset) +  Jrel_scaling_factor] = 1.0;
CONSTANTS[(208 * offset) +  Jup_b] = 1.0;
// CVAR: Additional scaling factor for Jleak and Jtr
CONSTANTS[(208 * offset) + Jtr_b] = 1.0;	// Trans_Total (NSR to JSR translocation)
CONSTANTS[(208 * offset) + Jleak_b] = 1.0;	// Leak_Total (Ca leak from NSR)
CONSTANTS[(208 * offset) +  frt] = CONSTANTS[(208 * offset) +  F]/( CONSTANTS[(208 * offset) +  R]*CONSTANTS[(208 * offset) +  T]);
CONSTANTS[(208 * offset) +  cmdnmax] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  cmdnmax_b]*1.30000 : CONSTANTS[(208 * offset) +  cmdnmax_b]);
CONSTANTS[(208 * offset) +  Ahs] = 1.00000 - CONSTANTS[(208 * offset) +  Ahf];
CONSTANTS[(208 * offset) +  thLp] =  3.00000*CONSTANTS[(208 * offset) +  thL];
CONSTANTS[(208 * offset) +  GNaL] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  GNaL_b]*0.600000 : CONSTANTS[(208 * offset) +  GNaL_b]);
CONSTANTS[(208 * offset) +  Gto] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  Gto_b]*4.00000 : CONSTANTS[(208 * offset) +  celltype]==2.00000 ?  CONSTANTS[(208 * offset) +  Gto_b]*4.00000 : CONSTANTS[(208 * offset) +  Gto_b]);
CONSTANTS[(208 * offset) +  Aff] = 0.600000;
CONSTANTS[(208 * offset) +  PCa] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  PCa_b]*1.20000 : CONSTANTS[(208 * offset) +  celltype]==2.00000 ?  CONSTANTS[(208 * offset) +  PCa_b]*2.50000 : CONSTANTS[(208 * offset) +  PCa_b]);
CONSTANTS[(208 * offset) +  tjca] = 75.0000;
CONSTANTS[(208 * offset) +  v0_CaL] = 0.000000;
CONSTANTS[(208 * offset) +  GKr] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  GKr_b]*1.30000 : CONSTANTS[(208 * offset) +  celltype]==2.00000 ?  CONSTANTS[(208 * offset) +  GKr_b]*0.800000 : CONSTANTS[(208 * offset) +  GKr_b]);
CONSTANTS[(208 * offset) +  GKs] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  GKs_b]*1.40000 : CONSTANTS[(208 * offset) +  GKs_b]);
CONSTANTS[(208 * offset) +  GK1] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  GK1_b]*1.20000 : CONSTANTS[(208 * offset) +  celltype]==2.00000 ?  CONSTANTS[(208 * offset) +  GK1_b]*1.30000 : CONSTANTS[(208 * offset) +  GK1_b]);
CONSTANTS[(208 * offset) +  vcell] =  1000.00*3.14000*CONSTANTS[(208 * offset) +  rad]*CONSTANTS[(208 * offset) +  rad]*CONSTANTS[(208 * offset) +  L];
CONSTANTS[(208 * offset) +  GKb] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  GKb_b]*0.600000 : CONSTANTS[(208 * offset) +  GKb_b]);
CONSTANTS[(208 * offset) +  v0_Nab] = 0.000000;
CONSTANTS[(208 * offset) +  v0_Cab] = 0.000000;
CONSTANTS[(208 * offset) +  a_rel] =  0.500000*CONSTANTS[(208 * offset) +  bt];
CONSTANTS[(208 * offset) +  btp] =  1.25000*CONSTANTS[(208 * offset) +  bt];
CONSTANTS[(208 * offset) +  upScale] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ? 1.30000 : 1.00000);
CONSTANTS[(208 * offset) +  cnc] = 0.000000;
CONSTANTS[(208 * offset) +  ffrt] =  CONSTANTS[(208 * offset) +  F]*CONSTANTS[(208 * offset) +  frt];
CONSTANTS[(208 * offset) +  Afs] = 1.00000 - CONSTANTS[(208 * offset) +  Aff];
CONSTANTS[(208 * offset) +  PCap] =  1.10000*CONSTANTS[(208 * offset) +  PCa];
CONSTANTS[(208 * offset) +  PCaNa] =  0.00125000*CONSTANTS[(208 * offset) +  PCa];
CONSTANTS[(208 * offset) +  PCaK] =  0.000357400*CONSTANTS[(208 * offset) +  PCa];
CONSTANTS[(208 * offset) +  B_1] =  2.00000*CONSTANTS[(208 * offset) +  frt];
CONSTANTS[(208 * offset) +  B_2] = CONSTANTS[(208 * offset) +  frt];
CONSTANTS[(208 * offset) +  B_3] = CONSTANTS[(208 * offset) +  frt];
CONSTANTS[(208 * offset) +  Ageo] =  2.00000*3.14000*CONSTANTS[(208 * offset) +  rad]*CONSTANTS[(208 * offset) +  rad]+ 2.00000*3.14000*CONSTANTS[(208 * offset) +  rad]*CONSTANTS[(208 * offset) +  L];
CONSTANTS[(208 * offset) +  B_Nab] = CONSTANTS[(208 * offset) +  frt];
CONSTANTS[(208 * offset) +  B_Cab] =  2.00000*CONSTANTS[(208 * offset) +  frt];
CONSTANTS[(208 * offset) +  a_relp] =  0.500000*CONSTANTS[(208 * offset) +  btp];
CONSTANTS[(208 * offset) +  PCaNap] =  0.00125000*CONSTANTS[(208 * offset) +  PCap];
CONSTANTS[(208 * offset) +  PCaKp] =  0.000357400*CONSTANTS[(208 * offset) +  PCap];
CONSTANTS[(208 * offset) +  Acap] =  2.00000*CONSTANTS[(208 * offset) +  Ageo];
CONSTANTS[(208 * offset) +  vmyo] =  0.680000*CONSTANTS[(208 * offset) +  vcell];
CONSTANTS[(208 * offset) +  vnsr] =  0.0552000*CONSTANTS[(208 * offset) +  vcell];
CONSTANTS[(208 * offset) +  vjsr] =  0.00480000*CONSTANTS[(208 * offset) +  vcell];
CONSTANTS[(208 * offset) +  vss] =  0.0200000*CONSTANTS[(208 * offset) +  vcell];
CONSTANTS[(208 * offset) +  h10_i] = CONSTANTS[(208 * offset) +  kasymm]+1.00000+ (CONSTANTS[(208 * offset) +  nao]/CONSTANTS[(208 * offset) +  kna1])*(1.00000+CONSTANTS[(208 * offset) +  nao]/CONSTANTS[(208 * offset) +  kna2]);
CONSTANTS[(208 * offset) +  h11_i] = ( CONSTANTS[(208 * offset) +  nao]*CONSTANTS[(208 * offset) +  nao])/( CONSTANTS[(208 * offset) +  h10_i]*CONSTANTS[(208 * offset) +  kna1]*CONSTANTS[(208 * offset) +  kna2]);
CONSTANTS[(208 * offset) +  h12_i] = 1.00000/CONSTANTS[(208 * offset) +  h10_i];
CONSTANTS[(208 * offset) +  k1_i] =  CONSTANTS[(208 * offset) +  h12_i]*CONSTANTS[(208 * offset) +  cao]*CONSTANTS[(208 * offset) +  kcaon];
CONSTANTS[(208 * offset) +  k2_i] = CONSTANTS[(208 * offset) +  kcaoff];
CONSTANTS[(208 * offset) +  k5_i] = CONSTANTS[(208 * offset) +  kcaoff];
CONSTANTS[(208 * offset) +  Gncx] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  Gncx_b]*1.10000 : CONSTANTS[(208 * offset) +  celltype]==2.00000 ?  CONSTANTS[(208 * offset) +  Gncx_b]*1.40000 : CONSTANTS[(208 * offset) +  Gncx_b]);
CONSTANTS[(208 * offset) +  h10_ss] = CONSTANTS[(208 * offset) +  kasymm]+1.00000+ (CONSTANTS[(208 * offset) +  nao]/CONSTANTS[(208 * offset) +  kna1])*(1.00000+CONSTANTS[(208 * offset) +  nao]/CONSTANTS[(208 * offset) +  kna2]);
CONSTANTS[(208 * offset) +  h11_ss] = ( CONSTANTS[(208 * offset) +  nao]*CONSTANTS[(208 * offset) +  nao])/( CONSTANTS[(208 * offset) +  h10_ss]*CONSTANTS[(208 * offset) +  kna1]*CONSTANTS[(208 * offset) +  kna2]);
CONSTANTS[(208 * offset) +  h12_ss] = 1.00000/CONSTANTS[(208 * offset) +  h10_ss];
CONSTANTS[(208 * offset) +  k1_ss] =  CONSTANTS[(208 * offset) +  h12_ss]*CONSTANTS[(208 * offset) +  cao]*CONSTANTS[(208 * offset) +  kcaon];
CONSTANTS[(208 * offset) +  k2_ss] = CONSTANTS[(208 * offset) +  kcaoff];
CONSTANTS[(208 * offset) +  k5_ss] = CONSTANTS[(208 * offset) +  kcaoff];
CONSTANTS[(208 * offset) +  b1] =  CONSTANTS[(208 * offset) +  k1m]*CONSTANTS[(208 * offset) +  MgADP];
CONSTANTS[(208 * offset) +  a2] = CONSTANTS[(208 * offset) +  k2p];
CONSTANTS[(208 * offset) +  a4] = (( CONSTANTS[(208 * offset) +  k4p]*CONSTANTS[(208 * offset) +  MgATP])/CONSTANTS[(208 * offset) +  Kmgatp])/(1.00000+CONSTANTS[(208 * offset) +  MgATP]/CONSTANTS[(208 * offset) +  Kmgatp]);
CONSTANTS[(208 * offset) +  Pnak] = (CONSTANTS[(208 * offset) +  celltype]==1.00000 ?  CONSTANTS[(208 * offset) +  Pnak_b]*0.900000 : CONSTANTS[(208 * offset) +  celltype]==2.00000 ?  CONSTANTS[(208 * offset) +  Pnak_b]*0.700000 : CONSTANTS[(208 * offset) +  Pnak_b]);
}

__device__ void applyDrugEffect(double *CONSTANTS, double conc, double *hill, int offset)
{
CONSTANTS[(208 * offset) + PCa_b] *= ((hill[(14 * offset) + 0] > 10E-14 && hill[(14 * offset) + 1] > 10E-14) ? 1./(1.+pow(conc/hill[(14 * offset) + 0],hill[(14 * offset) + 1])) : 1.);
CONSTANTS[(208 * offset) + GK1_b] *= ((hill[(14 * offset) + 2] > 10E-14 && hill[(14 * offset) + 3] > 10E-14) ? 1./(1.+pow(conc/hill[(14 * offset) + 2],hill[(14 * offset) + 3])) : 1.);
CONSTANTS[(208 * offset) + GKs_b] *= ((hill[(14 * offset) + 4] > 10E-14 && hill[(14 * offset) + 5] > 10E-14) ? 1./(1.+pow(conc/hill[(14 * offset) + 4],hill[(14 * offset) + 5])) : 1.);
CONSTANTS[(208 * offset) + GNa] *= ((hill[(14 * offset) + 6] > 10E-14 && hill[(14 * offset) + 7] > 10E-14) ? 1./(1.+pow(conc/hill[(14 * offset) + 6],hill[(14 * offset) + 7])) : 1.);
CONSTANTS[(208 * offset) + GNaL_b] *= ((hill[(14 * offset) + 8] > 10E-14 && hill[(14 * offset) + 9] > 10E-14) ? 1./(1.+pow(conc/hill[(14 * offset) + 8],hill[(14 * offset) + 9])) : 1.);
CONSTANTS[(208 * offset) + Gto_b] *= ((hill[(14 * offset) + 10] > 10E-14 && hill[(14 * offset) + 11] > 10E-14) ? 1./(1.+pow(conc/hill[(14 * offset) + 10],hill[(14 * offset) + 11])) : 1.);
//CONSTANTS[(208 * offset) + GKr_b] = CONSTANTS[GKr_b] * ((hill[(14 * offset) + 12] > 10E-14 && hill[(14 * offset) + 13] > 10E-14) ? 1./(1.+pow(conc/hill[(14 * offset) + 12],hill[(14 * offset) + 13])) : 1.);
}

__device__ void ___applyHERGBinding(double *CONSTANTS, double *STATES, double conc, double *herg, int offset)
{
CONSTANTS[(208 * offset) +  Kmax] = herg[(offset*6) + 0];
CONSTANTS[(208 * offset) +  Ku] = herg[(offset*6) + 1];
CONSTANTS[(208 * offset) +  n] = herg[(offset*6) + 2];
CONSTANTS[(208 * offset) +  halfmax] = herg[(offset*6) + 3];
CONSTANTS[(208 * offset) +  Vhalf] = herg[(offset*6) + 4];
CONSTANTS[(208 * offset) +  cnc] = conc;
STATES[(49 * offset) + D] = CONSTANTS[(208 * offset) +  cnc];
CONSTANTS[(208 * offset) +  Kt] = 0.000035;
}

__device__ void ___applyCvar(double *CONSTANTS, double *cvar, int offset)
{
  int num_of_constants= 208;

  CONSTANTS[(num_of_constants * offset) + GNa] *= cvar[(18 * offset) + 0];		// GNa
  CONSTANTS[(num_of_constants * offset) + GNaL_b] *= cvar[(18 * offset) + 1];		// GNaL
  CONSTANTS[(num_of_constants * offset) + Gto_b] *= cvar[(18 * offset) + 2];		// Gto
  CONSTANTS[(num_of_constants * offset) + GKr_b] *= cvar[(18 * offset) + 3];		// GKr
  CONSTANTS[(num_of_constants * offset) + GKs_b] *= cvar[(18 * offset) + 4];		// GKs
  CONSTANTS[(num_of_constants * offset) + GK1_b] *= cvar[(18 * offset) + 5];		// GK1
  CONSTANTS[(num_of_constants * offset) + Gncx_b] *= cvar[(18 * offset) + 6];		// GNaCa
  CONSTANTS[(num_of_constants * offset) + GKb_b] *= cvar[(18 * offset) + 7];		// GKb
  CONSTANTS[(num_of_constants * offset) + PCa_b] *= cvar[(18 * offset) + 8];		// PCa
  CONSTANTS[(num_of_constants * offset) + Pnak_b] *= cvar[(18 * offset) + 9];		// INaK
  CONSTANTS[(num_of_constants * offset) + PNab] *= cvar[(18 * offset) + 10];		// PNab
  CONSTANTS[(num_of_constants * offset) + PCab] *= cvar[(18 * offset) + 11];		// PCab
  CONSTANTS[(num_of_constants * offset) + GpCa] *= cvar[(18 * offset) + 12];		// GpCa
  CONSTANTS[(num_of_constants * offset) + KmCaMK] *= cvar[(18 * offset) + 17];	// KCaMK

  // Additional constants
  CONSTANTS[(num_of_constants * offset) + Jrel_scaling_factor] *= cvar[(18 * offset) + 13];	// SERCA_Total (release)
  CONSTANTS[(num_of_constants * offset) + Jup_b] *= cvar[(18 * offset) + 14];	// RyR_Total (uptake)
  CONSTANTS[(num_of_constants * offset) + Jtr_b] *= cvar[(18 * offset) + 15];	// Trans_Total (NSR to JSR translocation)
  CONSTANTS[(num_of_constants * offset) + Jleak_b] *= cvar[(18 * offset) + 16];	// Leak_Total (Ca leak from NSR)

}

__device__ void initConsts(double *CONSTANTS, double *STATES, double type, double conc, double *hill, double *herg, double *cvar, bool is_dutta, bool is_cvar, double bcl, double epsilon, int offset)
{
  ___initConsts(CONSTANTS, STATES, type, bcl, offset);

  if (offset == 0){
  printf("Celltype: %lf\n", CONSTANTS[celltype]);
  printf("Concentration: %lf\n", conc);
  printf("Control: \nPCa_b:%lf \nGK1_b:%lf \nGKs_b:%lf \nGNa:%lf \nGNaL_b:%lf \nGto_b:%lf \nGKr_b:%lf\n non b:\nPCa:%lf \nGK1:%lf \nGKs:%lf \nGNa:%lf \nGNaL:%lf \nGto:%lf \nGKr:%lf\n",
      CONSTANTS[PCa_b], CONSTANTS[GK1_b], CONSTANTS[GKs_b], CONSTANTS[GNa], CONSTANTS[GNaL_b], CONSTANTS[Gto_b], CONSTANTS[GKr_b],
      CONSTANTS[PCa], CONSTANTS[GK1], CONSTANTS[GKs], CONSTANTS[GNa], CONSTANTS[GNaL], CONSTANTS[Gto], CONSTANTS[GKr]);
  }

  if(is_cvar == true && offset == 0){  
    ___applyCvar(CONSTANTS, cvar, offset);
      printf("inter-individual data:\n");
  for(int idx = 0; idx < 18; idx++){
    printf("%lf,", cvar[idx]);
    }
    printf("\n");
    printf("After cvar: \nPCa:%lf \nGK1:%lf \nGKs:%lf \nGNa:%lf \nGNaL:%lf \nGto:%lf \nGKr:%lf\n",
      CONSTANTS[PCa], CONSTANTS[GK1], CONSTANTS[GKs], CONSTANTS[GNa], CONSTANTS[GNaL], CONSTANTS[Gto], CONSTANTS[GKr]);
	}

  applyDrugEffect(CONSTANTS, conc, hill, offset);
  
  if (offset == 0){
  printf("Hill data:\n");
  for(int idx = 0; idx < 14; idx++){
    printf("%lf,", hill[idx]);
    }
  printf("\n");
  printf("After drug: \nPCa:%lf \nGK1:%lf \nGKs:%lf \nGNa:%lf \nGNaL:%lf \nGto:%lf \nGKr:%lf\n",
      CONSTANTS[PCa_b], CONSTANTS[GK1_b], CONSTANTS[GKs_b], CONSTANTS[GNa], CONSTANTS[GNaL_b], CONSTANTS[Gto_b], CONSTANTS[GKr_b]);
  printf("Control hERG binding: \nKmax:%lf \nKu:%lf \nn:%lf \nhalfmax:%lf \nVhalf:%lf \nD:%lf \nKt:%lf\n", 
      CONSTANTS[Kmax], CONSTANTS[Ku], CONSTANTS[n], CONSTANTS[halfmax], CONSTANTS[Vhalf], STATES[D], CONSTANTS[Kt]);
  }

  if( conc > 10E-14 ) ___applyHERGBinding(CONSTANTS, STATES, conc, herg, offset);

  if (offset == 0){
  printf("hERG data:\n");
  for(int idx = 0; idx < 6; idx++){
    printf("%lf,", herg[idx]);
    }
  printf("\n");
  printf("Bootstraped hERG binding: \nKmax:%lf \nKu:%lf \nn:%lf \nhalfmax:%lf \nVhalf:%lf \nD:%lf \nKt:%lf\n",
      CONSTANTS[Kmax], CONSTANTS[Ku], CONSTANTS[n], CONSTANTS[halfmax], CONSTANTS[Vhalf], STATES[D], CONSTANTS[Kt]);
  }
  
}

__device__ void computeRates( double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC, int offset )
{
CONSTANTS[(208 * offset) + cmdnmax] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + cmdnmax_b]*1.30000 : CONSTANTS[(208 * offset) + cmdnmax_b]);
CONSTANTS[(208 * offset) + GNaL] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + GNaL_b]*0.600000 : CONSTANTS[(208 * offset) + GNaL_b]);
CONSTANTS[(208 * offset) + Gto] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + Gto_b]*4.00000 : CONSTANTS[(208 * offset) + celltype]==2.00000 ?  CONSTANTS[(208 * offset) + Gto_b]*4.00000 : CONSTANTS[(208 * offset) + Gto_b]);
CONSTANTS[(208 * offset) + PCa] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + PCa_b]*1.20000 : CONSTANTS[(208 * offset) + celltype]==2.00000 ?  CONSTANTS[(208 * offset) + PCa_b]*2.50000 : CONSTANTS[(208 * offset) + PCa_b]);
CONSTANTS[(208 * offset) + GKr] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + GKr_b]*1.30000 : CONSTANTS[(208 * offset) + celltype]==2.00000 ?  CONSTANTS[(208 * offset) + GKr_b]*0.800000 : CONSTANTS[(208 * offset) + GKr_b]);
CONSTANTS[(208 * offset) + GKs] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + GKs_b]*1.40000 : CONSTANTS[(208 * offset) + GKs_b]);
CONSTANTS[(208 * offset) + GK1] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + GK1_b]*1.20000 : CONSTANTS[(208 * offset) + celltype]==2.00000 ?  CONSTANTS[(208 * offset) + GK1_b]*1.30000 : CONSTANTS[(208 * offset) + GK1_b]);
CONSTANTS[(208 * offset) + GKb] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + GKb_b]*0.600000 : CONSTANTS[(208 * offset) + GKb_b]);
CONSTANTS[(208 * offset) + upScale] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ? 1.30000 : 1.00000);
CONSTANTS[(208 * offset) + Gncx] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + Gncx_b]*1.10000 : CONSTANTS[(208 * offset) + celltype]==2.00000 ?  CONSTANTS[(208 * offset) + Gncx_b]*1.40000 : CONSTANTS[(208 * offset) + Gncx_b]);
CONSTANTS[(208 * offset) + Pnak] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ?  CONSTANTS[(208 * offset) + Pnak_b]*0.900000 : CONSTANTS[(208 * offset) + celltype]==2.00000 ?  CONSTANTS[(208 * offset) + Pnak_b]*0.700000 : CONSTANTS[(208 * offset) + Pnak_b]);
// #ifdef TISSUE
// if(is_s1) ALGEBRAIC[Istim] = CONSTANTS[amp];
// else ALGEBRAIC[Istim] = 0.0;
// #else
ALGEBRAIC[(200 * offset) + Istim] = (TIME>=CONSTANTS[(208 * offset) + stim_start]&&TIME<=CONSTANTS[(208 * offset) + stim_end]&&(TIME - CONSTANTS[(208 * offset) + stim_start]) -  floor((TIME - CONSTANTS[(208 * offset) + stim_start])/CONSTANTS[(208 * offset) + BCL])*CONSTANTS[(208 * offset) + BCL]<=CONSTANTS[(208 * offset) + duration] ? CONSTANTS[(208 * offset) + amp] : 0.000000);
// #endif
ALGEBRAIC[(200 * offset) + hLss] = 1.00000/(1.00000+exp((STATES[(49 * offset) + V]+87.6100)/7.48800));
ALGEBRAIC[(200 * offset) + hLssp] = 1.00000/(1.00000+exp((STATES[(49 * offset) + V]+93.8100)/7.48800));
ALGEBRAIC[(200 * offset) + mss] = 1.00000/(1.00000+exp(- (STATES[(49 * offset) + V]+CONSTANTS[(208 * offset) + mssV1])/CONSTANTS[(208 * offset) + mssV2]));
ALGEBRAIC[(200 * offset) + tm] = 1.00000/( CONSTANTS[(208 * offset) + mtD1]*exp((STATES[(49 * offset) + V]+CONSTANTS[(208 * offset) + mtV1])/CONSTANTS[(208 * offset) + mtV2])+ CONSTANTS[(208 * offset) + mtD2]*exp(- (STATES[(49 * offset) + V]+CONSTANTS[(208 * offset) + mtV3])/CONSTANTS[(208 * offset) + mtV4]));
ALGEBRAIC[(200 * offset) + hss] = 1.00000/(1.00000+exp(((STATES[(49 * offset) + V]+CONSTANTS[(208 * offset) + hssV1]) - CONSTANTS[(208 * offset) + shift_INa_inact])/CONSTANTS[(208 * offset) + hssV2]));
ALGEBRAIC[(200 * offset) + thf] = 1.00000/( 1.43200e-05*exp(- ((STATES[(49 * offset) + V]+1.19600) - CONSTANTS[(208 * offset) + shift_INa_inact])/6.28500)+ 6.14900*exp(((STATES[(49 * offset) + V]+0.509600) - CONSTANTS[(208 * offset) + shift_INa_inact])/20.2700));
ALGEBRAIC[(200 * offset) + ths] = 1.00000/( 0.00979400*exp(- ((STATES[(49 * offset) + V]+17.9500) - CONSTANTS[(208 * offset) + shift_INa_inact])/28.0500)+ 0.334300*exp(((STATES[(49 * offset) + V]+5.73000) - CONSTANTS[(208 * offset) + shift_INa_inact])/56.6600));
ALGEBRAIC[(200 * offset) + ass] = 1.00000/(1.00000+exp(- (STATES[(49 * offset) + V] - 14.3400)/14.8200));
ALGEBRAIC[(200 * offset) + ta] = 1.05150/(1.00000/( 1.20890*(1.00000+exp(- (STATES[(49 * offset) + V] - 18.4099)/29.3814)))+3.50000/(1.00000+exp((STATES[(49 * offset) + V]+100.000)/29.3814)));
ALGEBRAIC[(200 * offset) + dss] = 1.00000/(1.00000+exp(- (STATES[(49 * offset) + V]+3.94000)/4.23000));
ALGEBRAIC[(200 * offset) + td] = 0.600000+1.00000/(exp( - 0.0500000*(STATES[(49 * offset) + V]+6.00000))+exp( 0.0900000*(STATES[(49 * offset) + V]+14.0000)));
ALGEBRAIC[(200 * offset) + fss] = 1.00000/(1.00000+exp((STATES[(49 * offset) + V]+19.5800)/3.69600));
ALGEBRAIC[(200 * offset) + tff] = 7.00000+1.00000/( 0.00450000*exp(- (STATES[(49 * offset) + V]+20.0000)/10.0000)+ 0.00450000*exp((STATES[(49 * offset) + V]+20.0000)/10.0000));
ALGEBRAIC[(200 * offset) + tfs] = 1000.00+1.00000/( 3.50000e-05*exp(- (STATES[(49 * offset) + V]+5.00000)/4.00000)+ 3.50000e-05*exp((STATES[(49 * offset) + V]+5.00000)/6.00000));
ALGEBRAIC[(200 * offset) + fcass] = ALGEBRAIC[(200 * offset) + fss];
ALGEBRAIC[(200 * offset) + km2n] =  STATES[(49 * offset) + jca]*1.00000;
ALGEBRAIC[(200 * offset) + anca] = 1.00000/(CONSTANTS[(208 * offset) + k2n]/ALGEBRAIC[(200 * offset) + km2n]+pow(1.00000+CONSTANTS[(208 * offset) + Kmn]/STATES[(49 * offset) + cass], 4.00000));
ALGEBRAIC[(200 * offset) + xs1ss] = 1.00000/(1.00000+exp(- (STATES[(49 * offset) + V]+11.6000)/8.93200));
ALGEBRAIC[(200 * offset) + txs1] = CONSTANTS[(208 * offset) + txs1_max]+1.00000/( 0.000232600*exp((STATES[(49 * offset) + V]+48.2800)/17.8000)+ 0.00129200*exp(- (STATES[(49 * offset) + V]+210.000)/230.000));
ALGEBRAIC[(200 * offset) + xk1ss] = 1.00000/(1.00000+exp(- (STATES[(49 * offset) + V]+ 2.55380*CONSTANTS[(208 * offset) + ko]+144.590)/( 1.56920*CONSTANTS[(208 * offset) + ko]+3.81150)));
ALGEBRAIC[(200 * offset) + txk1] = 122.200/(exp(- (STATES[(49 * offset) + V]+127.200)/20.3600)+exp((STATES[(49 * offset) + V]+236.800)/69.3300));
ALGEBRAIC[(200 * offset) + CaMKb] = ( CONSTANTS[(208 * offset) + CaMKo]*(1.00000 - STATES[(49 * offset) + CaMKt]))/(1.00000+CONSTANTS[(208 * offset) + KmCaM]/STATES[(49 * offset) + cass]);
ALGEBRAIC[(200 * offset) + jss] = ALGEBRAIC[(200 * offset) + hss];
ALGEBRAIC[(200 * offset) + tj] = 2.03800+1.00000/( 0.0213600*exp(- ((STATES[(49 * offset) + V]+100.600) - CONSTANTS[(208 * offset) + shift_INa_inact])/8.28100)+ 0.305200*exp(((STATES[(49 * offset) + V]+0.994100) - CONSTANTS[(208 * offset) + shift_INa_inact])/38.4500));
ALGEBRAIC[(200 * offset) + assp] = 1.00000/(1.00000+exp(- (STATES[(49 * offset) + V] - 24.3400)/14.8200));
ALGEBRAIC[(200 * offset) + tfcaf] = 7.00000+1.00000/( 0.0400000*exp(- (STATES[(49 * offset) + V] - 4.00000)/7.00000)+ 0.0400000*exp((STATES[(49 * offset) + V] - 4.00000)/7.00000));
ALGEBRAIC[(200 * offset) + tfcas] = 100.000+1.00000/( 0.000120000*exp(- STATES[(49 * offset) + V]/3.00000)+ 0.000120000*exp(STATES[(49 * offset) + V]/7.00000));
ALGEBRAIC[(200 * offset) + tffp] =  2.50000*ALGEBRAIC[(200 * offset) + tff];
ALGEBRAIC[(200 * offset) + xs2ss] = ALGEBRAIC[(200 * offset) + xs1ss];
ALGEBRAIC[(200 * offset) + txs2] = 1.00000/( 0.0100000*exp((STATES[(49 * offset) + V] - 50.0000)/20.0000)+ 0.0193000*exp(- (STATES[(49 * offset) + V]+66.5400)/31.0000));
ALGEBRAIC[(200 * offset) + hssp] = 1.00000/(1.00000+exp(((STATES[(49 * offset) + V]+89.1000) - CONSTANTS[(208 * offset) + shift_INa_inact])/6.08600));
ALGEBRAIC[(200 * offset) + thsp] =  3.00000*ALGEBRAIC[(200 * offset) + ths];
ALGEBRAIC[(200 * offset) + tjp] =  1.46000*ALGEBRAIC[(200 * offset) + tj];
ALGEBRAIC[(200 * offset) + mLss] = 1.00000/(1.00000+exp(- (STATES[(49 * offset) + V]+42.8500)/5.26400));
ALGEBRAIC[(200 * offset) + tmL] = ALGEBRAIC[(200 * offset) + tm];
ALGEBRAIC[(200 * offset) + tfcafp] =  2.50000*ALGEBRAIC[(200 * offset) + tfcaf];
ALGEBRAIC[(200 * offset) + iss] = 1.00000/(1.00000+exp((STATES[(49 * offset) + V]+43.9400)/5.71100));
ALGEBRAIC[(200 * offset) + delta_epi] = (CONSTANTS[(208 * offset) + celltype]==1.00000 ? 1.00000 - 0.950000/(1.00000+exp((STATES[(49 * offset) + V]+70.0000)/5.00000)) : 1.00000);
ALGEBRAIC[(200 * offset) + tiF_b] = 4.56200+1.00000/( 0.393300*exp(- (STATES[(49 * offset) + V]+100.000)/100.000)+ 0.0800400*exp((STATES[(49 * offset) + V]+50.0000)/16.5900));
ALGEBRAIC[(200 * offset) + tiF] =  ALGEBRAIC[(200 * offset) + tiF_b]*ALGEBRAIC[(200 * offset) + delta_epi];
ALGEBRAIC[(200 * offset) + tiS_b] = 23.6200+1.00000/( 0.00141600*exp(- (STATES[(49 * offset) + V]+96.5200)/59.0500)+ 1.78000e-08*exp((STATES[(49 * offset) + V]+114.100)/8.07900));
ALGEBRAIC[(200 * offset) + tiS] =  ALGEBRAIC[(200 * offset) + tiS_b]*ALGEBRAIC[(200 * offset) + delta_epi];
ALGEBRAIC[(200 * offset) + dti_develop] = 1.35400+0.000100000/(exp((STATES[(49 * offset) + V] - 167.400)/15.8900)+exp(- (STATES[(49 * offset) + V] - 12.2300)/0.215400));
ALGEBRAIC[(200 * offset) + dti_recover] = 1.00000 - 0.500000/(1.00000+exp((STATES[(49 * offset) + V]+70.0000)/20.0000));
ALGEBRAIC[(200 * offset) + tiFp] =  ALGEBRAIC[(200 * offset) + dti_develop]*ALGEBRAIC[(200 * offset) + dti_recover]*ALGEBRAIC[(200 * offset) + tiF];
ALGEBRAIC[(200 * offset) + tiSp] =  ALGEBRAIC[(200 * offset) + dti_develop]*ALGEBRAIC[(200 * offset) + dti_recover]*ALGEBRAIC[(200 * offset) + tiS];
ALGEBRAIC[(200 * offset) + f] =  CONSTANTS[(208 * offset) + Aff]*STATES[(49 * offset) + ff]+ CONSTANTS[(208 * offset) + Afs]*STATES[(49 * offset) + fs];
ALGEBRAIC[(200 * offset) + Afcaf] = 0.300000+0.600000/(1.00000+exp((STATES[(49 * offset) + V] - 10.0000)/10.0000));
ALGEBRAIC[(200 * offset) + Afcas] = 1.00000 - ALGEBRAIC[(200 * offset) + Afcaf];
ALGEBRAIC[(200 * offset) + fca] =  ALGEBRAIC[(200 * offset) + Afcaf]*STATES[(49 * offset) + fcaf]+ ALGEBRAIC[(200 * offset) + Afcas]*STATES[(49 * offset) + fcas];
ALGEBRAIC[(200 * offset) + fp] =  CONSTANTS[(208 * offset) + Aff]*STATES[(49 * offset) + ffp]+ CONSTANTS[(208 * offset) + Afs]*STATES[(49 * offset) + fs];
ALGEBRAIC[(200 * offset) + fcap] =  ALGEBRAIC[(200 * offset) + Afcaf]*STATES[(49 * offset) + fcafp]+ ALGEBRAIC[(200 * offset) + Afcas]*STATES[(49 * offset) + fcas];
ALGEBRAIC[(200 * offset) + vfrt] =  STATES[(49 * offset) + V]*CONSTANTS[(208 * offset) + frt];
ALGEBRAIC[(200 * offset) + A_1] = ( 4.00000*CONSTANTS[(208 * offset) + ffrt]*( STATES[(49 * offset) + cass]*exp( 2.00000*ALGEBRAIC[(200 * offset) + vfrt]) -  0.341000*CONSTANTS[(208 * offset) + cao]))/CONSTANTS[(208 * offset) + B_1];
ALGEBRAIC[(200 * offset) + U_1] =  CONSTANTS[(208 * offset) + B_1]*(STATES[(49 * offset) + V] - CONSTANTS[(208 * offset) + v0_CaL]);
ALGEBRAIC[(200 * offset) + PhiCaL] = (- 1.00000e-07<=ALGEBRAIC[(200 * offset) + U_1]&&ALGEBRAIC[(200 * offset) + U_1]<=1.00000e-07 ?  ALGEBRAIC[(200 * offset) + A_1]*(1.00000 -  0.500000*ALGEBRAIC[(200 * offset) + U_1]) : ( ALGEBRAIC[(200 * offset) + A_1]*ALGEBRAIC[(200 * offset) + U_1])/(exp(ALGEBRAIC[(200 * offset) + U_1]) - 1.00000));
ALGEBRAIC[(200 * offset) + CaMKa] = ALGEBRAIC[(200 * offset) + CaMKb]+STATES[(49 * offset) + CaMKt];
ALGEBRAIC[(200 * offset) + fICaLp] = 1.00000/(1.00000+CONSTANTS[(208 * offset) + KmCaMK]/ALGEBRAIC[(200 * offset) + CaMKa]);
ALGEBRAIC[(200 * offset) + ICaL] =  (1.00000 - ALGEBRAIC[(200 * offset) + fICaLp])*CONSTANTS[(208 * offset) + PCa]*ALGEBRAIC[(200 * offset) + PhiCaL]*STATES[(49 * offset) + d]*( ALGEBRAIC[(200 * offset) + f]*(1.00000 - STATES[(49 * offset) + nca])+ STATES[(49 * offset) + jca]*ALGEBRAIC[(200 * offset) + fca]*STATES[(49 * offset) + nca])+ ALGEBRAIC[(200 * offset) + fICaLp]*CONSTANTS[(208 * offset) + PCap]*ALGEBRAIC[(200 * offset) + PhiCaL]*STATES[(49 * offset) + d]*( ALGEBRAIC[(200 * offset) + fp]*(1.00000 - STATES[(49 * offset) + nca])+ STATES[(49 * offset) + jca]*ALGEBRAIC[(200 * offset) + fcap]*STATES[(49 * offset) + nca]);
ALGEBRAIC[(200 * offset) + Jrel_inf_temp] = ( CONSTANTS[(208 * offset) + a_rel]*- ALGEBRAIC[(200 * offset) + ICaL])/(1.00000+ 1.00000*pow(1.50000/STATES[(49 * offset) + cajsr], 8.00000));
ALGEBRAIC[(200 * offset) + Jrel_inf] = (CONSTANTS[(208 * offset) + celltype]==2.00000 ?  ALGEBRAIC[(200 * offset) + Jrel_inf_temp]*1.70000 : ALGEBRAIC[(200 * offset) + Jrel_inf_temp]);
ALGEBRAIC[(200 * offset) + tau_rel_temp] = CONSTANTS[(208 * offset) + bt]/(1.00000+0.0123000/STATES[(49 * offset) + cajsr]);
ALGEBRAIC[(200 * offset) + tau_rel] = (ALGEBRAIC[(200 * offset) + tau_rel_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(200 * offset) + tau_rel_temp]);
ALGEBRAIC[(200 * offset) + Jrel_temp] = ( CONSTANTS[(208 * offset) + a_relp]*- ALGEBRAIC[(200 * offset) + ICaL])/(1.00000+pow(1.50000/STATES[(49 * offset) + cajsr], 8.00000));
ALGEBRAIC[(200 * offset) + Jrel_infp] = (CONSTANTS[(208 * offset) + celltype]==2.00000 ?  ALGEBRAIC[(200 * offset) + Jrel_temp]*1.70000 : ALGEBRAIC[(200 * offset) + Jrel_temp]);
ALGEBRAIC[(200 * offset) + tau_relp_temp] = CONSTANTS[(208 * offset) + btp]/(1.00000+0.0123000/STATES[(49 * offset) + cajsr]);
ALGEBRAIC[(200 * offset) + tau_relp] = (ALGEBRAIC[(200 * offset) + tau_relp_temp]<0.00100000 ? 0.00100000 : ALGEBRAIC[(200 * offset) + tau_relp_temp]);
ALGEBRAIC[(200 * offset) + EK] =  (( CONSTANTS[(208 * offset) + R]*CONSTANTS[(208 * offset) + T])/CONSTANTS[(208 * offset) + F])*log(CONSTANTS[(208 * offset) + ko]/STATES[(49 * offset) + ki]);
ALGEBRAIC[(200 * offset) + AiF] = 1.00000/(1.00000+exp((STATES[(49 * offset) + V] - 213.600)/151.200));
ALGEBRAIC[(200 * offset) + AiS] = 1.00000 - ALGEBRAIC[(200 * offset) + AiF];
ALGEBRAIC[(200 * offset) + i] =  ALGEBRAIC[(200 * offset) + AiF]*STATES[(49 * offset) + iF]+ ALGEBRAIC[(200 * offset) + AiS]*STATES[(49 * offset) + iS];
ALGEBRAIC[(200 * offset) + ip] =  ALGEBRAIC[(200 * offset) + AiF]*STATES[(49 * offset) + iFp]+ ALGEBRAIC[(200 * offset) + AiS]*STATES[(49 * offset) + iSp];
ALGEBRAIC[(200 * offset) + fItop] = 1.00000/(1.00000+CONSTANTS[(208 * offset) + KmCaMK]/ALGEBRAIC[(200 * offset) + CaMKa]);
ALGEBRAIC[(200 * offset) + Ito] =  CONSTANTS[(208 * offset) + Gto]*(STATES[(49 * offset) + V] - ALGEBRAIC[(200 * offset) + EK])*( (1.00000 - ALGEBRAIC[(200 * offset) + fItop])*STATES[(49 * offset) + a]*ALGEBRAIC[(200 * offset) + i]+ ALGEBRAIC[(200 * offset) + fItop]*STATES[(49 * offset) + ap]*ALGEBRAIC[(200 * offset) + ip]);
ALGEBRAIC[(200 * offset) + IKr] =  CONSTANTS[(208 * offset) + GKr]* pow((CONSTANTS[(208 * offset) + ko]/5.40000), 1.0 / 2)*STATES[(49 * offset) + O]*(STATES[(49 * offset) + V] - ALGEBRAIC[(200 * offset) + EK]);
ALGEBRAIC[(200 * offset) + EKs] =  (( CONSTANTS[(208 * offset) + R]*CONSTANTS[(208 * offset) + T])/CONSTANTS[(208 * offset) + F])*log((CONSTANTS[(208 * offset) + ko]+ CONSTANTS[(208 * offset) + PKNa]*CONSTANTS[(208 * offset) + nao])/(STATES[(49 * offset) + ki]+ CONSTANTS[(208 * offset) + PKNa]*STATES[(49 * offset) + nai]));
ALGEBRAIC[(200 * offset) + KsCa] = 1.00000+0.600000/(1.00000+pow(3.80000e-05/STATES[(49 * offset) + cai], 1.40000));
ALGEBRAIC[(200 * offset) + IKs] =  CONSTANTS[(208 * offset) + GKs]*ALGEBRAIC[(200 * offset) + KsCa]*STATES[(49 * offset) + xs1]*STATES[(49 * offset) + xs2]*(STATES[(49 * offset) + V] - ALGEBRAIC[(200 * offset) + EKs]);
ALGEBRAIC[(200 * offset) + rk1] = 1.00000/(1.00000+exp(((STATES[(49 * offset) + V]+105.800) -  2.60000*CONSTANTS[(208 * offset) + ko])/9.49300));
ALGEBRAIC[(200 * offset) + IK1] =  CONSTANTS[(208 * offset) + GK1]* pow(CONSTANTS[(208 * offset) + ko], 1.0 / 2)*ALGEBRAIC[(200 * offset) + rk1]*STATES[(49 * offset) + xk1]*(STATES[(49 * offset) + V] - ALGEBRAIC[(200 * offset) + EK]);
ALGEBRAIC[(200 * offset) + Knao] =  CONSTANTS[(208 * offset) + Knao0]*exp(( (1.00000 - CONSTANTS[(208 * offset) + delta])*STATES[(49 * offset) + V]*CONSTANTS[(208 * offset) + F])/( 3.00000*CONSTANTS[(208 * offset) + R]*CONSTANTS[(208 * offset) + T]));
ALGEBRAIC[(200 * offset) + a3] = ( CONSTANTS[(208 * offset) + k3p]*pow(CONSTANTS[(208 * offset) + ko]/CONSTANTS[(208 * offset) + Kko], 2.00000))/((pow(1.00000+CONSTANTS[(208 * offset) + nao]/ALGEBRAIC[(200 * offset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(208 * offset) + ko]/CONSTANTS[(208 * offset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(200 * offset) + P] = CONSTANTS[(208 * offset) + eP]/(1.00000+CONSTANTS[(208 * offset) + H]/CONSTANTS[(208 * offset) + Khp]+STATES[(49 * offset) + nai]/CONSTANTS[(208 * offset) + Knap]+STATES[(49 * offset) + ki]/CONSTANTS[(208 * offset) + Kxkur]);
ALGEBRAIC[(200 * offset) + b3] = ( CONSTANTS[(208 * offset) + k3m]*ALGEBRAIC[(200 * offset) + P]*CONSTANTS[(208 * offset) + H])/(1.00000+CONSTANTS[(208 * offset) + MgATP]/CONSTANTS[(208 * offset) + Kmgatp]);
ALGEBRAIC[(200 * offset) + Knai] =  CONSTANTS[(208 * offset) + Knai0]*exp(( CONSTANTS[(208 * offset) + delta]*STATES[(49 * offset) + V]*CONSTANTS[(208 * offset) + F])/( 3.00000*CONSTANTS[(208 * offset) + R]*CONSTANTS[(208 * offset) + T]));
ALGEBRAIC[(200 * offset) + a1] = ( CONSTANTS[(208 * offset) + k1p]*pow(STATES[(49 * offset) + nai]/ALGEBRAIC[(200 * offset) + Knai], 3.00000))/((pow(1.00000+STATES[(49 * offset) + nai]/ALGEBRAIC[(200 * offset) + Knai], 3.00000)+pow(1.00000+STATES[(49 * offset) + ki]/CONSTANTS[(208 * offset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(200 * offset) + b2] = ( CONSTANTS[(208 * offset) + k2m]*pow(CONSTANTS[(208 * offset) + nao]/ALGEBRAIC[(200 * offset) + Knao], 3.00000))/((pow(1.00000+CONSTANTS[(208 * offset) + nao]/ALGEBRAIC[(200 * offset) + Knao], 3.00000)+pow(1.00000+CONSTANTS[(208 * offset) + ko]/CONSTANTS[(208 * offset) + Kko], 2.00000)) - 1.00000);
ALGEBRAIC[(200 * offset) + b4] = ( CONSTANTS[(208 * offset) + k4m]*pow(STATES[(49 * offset) + ki]/CONSTANTS[(208 * offset) + Kki], 2.00000))/((pow(1.00000+STATES[(49 * offset) + nai]/ALGEBRAIC[(200 * offset) + Knai], 3.00000)+pow(1.00000+STATES[(49 * offset) + ki]/CONSTANTS[(208 * offset) + Kki], 2.00000)) - 1.00000);
ALGEBRAIC[(200 * offset) + x1] =  CONSTANTS[(208 * offset) + a4]*ALGEBRAIC[(200 * offset) + a1]*CONSTANTS[(208 * offset) + a2]+ ALGEBRAIC[(200 * offset) + b2]*ALGEBRAIC[(200 * offset) + b4]*ALGEBRAIC[(200 * offset) + b3]+ CONSTANTS[(208 * offset) + a2]*ALGEBRAIC[(200 * offset) + b4]*ALGEBRAIC[(200 * offset) + b3]+ ALGEBRAIC[(200 * offset) + b3]*ALGEBRAIC[(200 * offset) + a1]*CONSTANTS[(208 * offset) + a2];
ALGEBRAIC[(200 * offset) + x2] =  ALGEBRAIC[(200 * offset) + b2]*CONSTANTS[(208 * offset) + b1]*ALGEBRAIC[(200 * offset) + b4]+ ALGEBRAIC[(200 * offset) + a1]*CONSTANTS[(208 * offset) + a2]*ALGEBRAIC[(200 * offset) + a3]+ ALGEBRAIC[(200 * offset) + a3]*CONSTANTS[(208 * offset) + b1]*ALGEBRAIC[(200 * offset) + b4]+ CONSTANTS[(208 * offset) + a2]*ALGEBRAIC[(200 * offset) + a3]*ALGEBRAIC[(200 * offset) + b4];
ALGEBRAIC[(200 * offset) + x3] =  CONSTANTS[(208 * offset) + a2]*ALGEBRAIC[(200 * offset) + a3]*CONSTANTS[(208 * offset) + a4]+ ALGEBRAIC[(200 * offset) + b3]*ALGEBRAIC[(200 * offset) + b2]*CONSTANTS[(208 * offset) + b1]+ ALGEBRAIC[(200 * offset) + b2]*CONSTANTS[(208 * offset) + b1]*CONSTANTS[(208 * offset) + a4]+ ALGEBRAIC[(200 * offset) + a3]*CONSTANTS[(208 * offset) + a4]*CONSTANTS[(208 * offset) + b1];
ALGEBRAIC[(200 * offset) + x4] =  ALGEBRAIC[(200 * offset) + b4]*ALGEBRAIC[(200 * offset) + b3]*ALGEBRAIC[(200 * offset) + b2]+ ALGEBRAIC[(200 * offset) + a3]*CONSTANTS[(208 * offset) + a4]*ALGEBRAIC[(200 * offset) + a1]+ ALGEBRAIC[(200 * offset) + b2]*CONSTANTS[(208 * offset) + a4]*ALGEBRAIC[(200 * offset) + a1]+ ALGEBRAIC[(200 * offset) + b3]*ALGEBRAIC[(200 * offset) + b2]*ALGEBRAIC[(200 * offset) + a1];
ALGEBRAIC[(200 * offset) + E1] = ALGEBRAIC[(200 * offset) + x1]/(ALGEBRAIC[(200 * offset) + x1]+ALGEBRAIC[(200 * offset) + x2]+ALGEBRAIC[(200 * offset) + x3]+ALGEBRAIC[(200 * offset) + x4]);
ALGEBRAIC[(200 * offset) + E2] = ALGEBRAIC[(200 * offset) + x2]/(ALGEBRAIC[(200 * offset) + x1]+ALGEBRAIC[(200 * offset) + x2]+ALGEBRAIC[(200 * offset) + x3]+ALGEBRAIC[(200 * offset) + x4]);
ALGEBRAIC[(200 * offset) + JnakNa] =  3.00000*( ALGEBRAIC[(200 * offset) + E1]*ALGEBRAIC[(200 * offset) + a3] -  ALGEBRAIC[(200 * offset) + E2]*ALGEBRAIC[(200 * offset) + b3]);
ALGEBRAIC[(200 * offset) + E3] = ALGEBRAIC[(200 * offset) + x3]/(ALGEBRAIC[(200 * offset) + x1]+ALGEBRAIC[(200 * offset) + x2]+ALGEBRAIC[(200 * offset) + x3]+ALGEBRAIC[(200 * offset) + x4]);
ALGEBRAIC[(200 * offset) + E4] = ALGEBRAIC[(200 * offset) + x4]/(ALGEBRAIC[(200 * offset) + x1]+ALGEBRAIC[(200 * offset) + x2]+ALGEBRAIC[(200 * offset) + x3]+ALGEBRAIC[(200 * offset) + x4]);
ALGEBRAIC[(200 * offset) + JnakK] =  2.00000*( ALGEBRAIC[(200 * offset) + E4]*CONSTANTS[(208 * offset) + b1] -  ALGEBRAIC[(200 * offset) + E3]*ALGEBRAIC[(200 * offset) + a1]);
ALGEBRAIC[(200 * offset) + INaK] =  CONSTANTS[(208 * offset) + Pnak]*( CONSTANTS[(208 * offset) + zna]*ALGEBRAIC[(200 * offset) + JnakNa]+ CONSTANTS[(208 * offset) + zk]*ALGEBRAIC[(200 * offset) + JnakK]);
ALGEBRAIC[(200 * offset) + xkb] = 1.00000/(1.00000+exp(- (STATES[(49 * offset) + V] - 14.4800)/18.3400));
ALGEBRAIC[(200 * offset) + IKb] =  CONSTANTS[(208 * offset) + GKb]*ALGEBRAIC[(200 * offset) + xkb]*(STATES[(49 * offset) + V] - ALGEBRAIC[(200 * offset) + EK]);
ALGEBRAIC[(200 * offset) + JdiffK] = (STATES[(49 * offset) + kss] - STATES[(49 * offset) + ki])/2.00000;
ALGEBRAIC[(200 * offset) + A_3] = ( 0.750000*CONSTANTS[(208 * offset) + ffrt]*( STATES[(49 * offset) + kss]*exp(ALGEBRAIC[(200 * offset) + vfrt]) - CONSTANTS[(208 * offset) + ko]))/CONSTANTS[(208 * offset) + B_3];
ALGEBRAIC[(200 * offset) + U_3] =  CONSTANTS[(208 * offset) + B_3]*(STATES[(49 * offset) + V] - CONSTANTS[(208 * offset) + v0_CaL]);
ALGEBRAIC[(200 * offset) + PhiCaK] = (- 1.00000e-07<=ALGEBRAIC[(200 * offset) + U_3]&&ALGEBRAIC[(200 * offset) + U_3]<=1.00000e-07 ?  ALGEBRAIC[(200 * offset) + A_3]*(1.00000 -  0.500000*ALGEBRAIC[(200 * offset) + U_3]) : ( ALGEBRAIC[(200 * offset) + A_3]*ALGEBRAIC[(200 * offset) + U_3])/(exp(ALGEBRAIC[(200 * offset) + U_3]) - 1.00000));
ALGEBRAIC[(200 * offset) + ICaK] =  (1.00000 - ALGEBRAIC[(200 * offset) + fICaLp])*CONSTANTS[(208 * offset) + PCaK]*ALGEBRAIC[(200 * offset) + PhiCaK]*STATES[(49 * offset) + d]*( ALGEBRAIC[(200 * offset) + f]*(1.00000 - STATES[(49 * offset) + nca])+ STATES[(49 * offset) + jca]*ALGEBRAIC[(200 * offset) + fca]*STATES[(49 * offset) + nca])+ ALGEBRAIC[(200 * offset) + fICaLp]*CONSTANTS[(208 * offset) + PCaKp]*ALGEBRAIC[(200 * offset) + PhiCaK]*STATES[(49 * offset) + d]*( ALGEBRAIC[(200 * offset) + fp]*(1.00000 - STATES[(49 * offset) + nca])+ STATES[(49 * offset) + jca]*ALGEBRAIC[(200 * offset) + fcap]*STATES[(49 * offset) + nca]);
ALGEBRAIC[(200 * offset) + ENa] =  (( CONSTANTS[(208 * offset) + R]*CONSTANTS[(208 * offset) + T])/CONSTANTS[(208 * offset) + F])*log(CONSTANTS[(208 * offset) + nao]/STATES[(49 * offset) + nai]);
ALGEBRAIC[(200 * offset) + h] =  CONSTANTS[(208 * offset) + Ahf]*STATES[(49 * offset) + hf]+ CONSTANTS[(208 * offset) + Ahs]*STATES[(49 * offset) + hs];
ALGEBRAIC[(200 * offset) + hp] =  CONSTANTS[(208 * offset) + Ahf]*STATES[(49 * offset) + hf]+ CONSTANTS[(208 * offset) + Ahs]*STATES[(49 * offset) + hsp];
ALGEBRAIC[(200 * offset) + fINap] = 1.00000/(1.00000+CONSTANTS[(208 * offset) + KmCaMK]/ALGEBRAIC[(200 * offset) + CaMKa]);
ALGEBRAIC[(200 * offset) + INa] =  CONSTANTS[(208 * offset) + GNa]*(STATES[(49 * offset) + V] - ALGEBRAIC[(200 * offset) + ENa])*pow(STATES[(49 * offset) + m], 3.00000)*( (1.00000 - ALGEBRAIC[(200 * offset) + fINap])*ALGEBRAIC[(200 * offset) + h]*STATES[(49 * offset) + j]+ ALGEBRAIC[(200 * offset) + fINap]*ALGEBRAIC[(200 * offset) + hp]*STATES[(49 * offset) + jp]);
ALGEBRAIC[(200 * offset) + fINaLp] = 1.00000/(1.00000+CONSTANTS[(208 * offset) + KmCaMK]/ALGEBRAIC[(200 * offset) + CaMKa]);
ALGEBRAIC[(200 * offset) + INaL] =  CONSTANTS[(208 * offset) + GNaL]*(STATES[(49 * offset) + V] - ALGEBRAIC[(200 * offset) + ENa])*STATES[(49 * offset) + mL]*( (1.00000 - ALGEBRAIC[(200 * offset) + fINaLp])*STATES[(49 * offset) + hL]+ ALGEBRAIC[(200 * offset) + fINaLp]*STATES[(49 * offset) + hLp]);
ALGEBRAIC[(200 * offset) + allo_i] = 1.00000/(1.00000+pow(CONSTANTS[(208 * offset) + KmCaAct]/STATES[(49 * offset) + cai], 2.00000));
ALGEBRAIC[(200 * offset) + hna] = exp(( CONSTANTS[(208 * offset) + qna]*STATES[(49 * offset) + V]*CONSTANTS[(208 * offset) + F])/( CONSTANTS[(208 * offset) + R]*CONSTANTS[(208 * offset) + T]));
ALGEBRAIC[(200 * offset) + h7_i] = 1.00000+ (CONSTANTS[(208 * offset) + nao]/CONSTANTS[(208 * offset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(200 * offset) + hna]);
ALGEBRAIC[(200 * offset) + h8_i] = CONSTANTS[(208 * offset) + nao]/( CONSTANTS[(208 * offset) + kna3]*ALGEBRAIC[(200 * offset) + hna]*ALGEBRAIC[(200 * offset) + h7_i]);
ALGEBRAIC[(200 * offset) + k3pp_i] =  ALGEBRAIC[(200 * offset) + h8_i]*CONSTANTS[(208 * offset) + wnaca];
ALGEBRAIC[(200 * offset) + h1_i] = 1.00000+ (STATES[(49 * offset) + nai]/CONSTANTS[(208 * offset) + kna3])*(1.00000+ALGEBRAIC[(200 * offset) + hna]);
ALGEBRAIC[(200 * offset) + h2_i] = ( STATES[(49 * offset) + nai]*ALGEBRAIC[(200 * offset) + hna])/( CONSTANTS[(208 * offset) + kna3]*ALGEBRAIC[(200 * offset) + h1_i]);
ALGEBRAIC[(200 * offset) + k4pp_i] =  ALGEBRAIC[(200 * offset) + h2_i]*CONSTANTS[(208 * offset) + wnaca];
ALGEBRAIC[(200 * offset) + h4_i] = 1.00000+ (STATES[(49 * offset) + nai]/CONSTANTS[(208 * offset) + kna1])*(1.00000+STATES[(49 * offset) + nai]/CONSTANTS[(208 * offset) + kna2]);
ALGEBRAIC[(200 * offset) + h5_i] = ( STATES[(49 * offset) + nai]*STATES[(49 * offset) + nai])/( ALGEBRAIC[(200 * offset) + h4_i]*CONSTANTS[(208 * offset) + kna1]*CONSTANTS[(208 * offset) + kna2]);
ALGEBRAIC[(200 * offset) + k7_i] =  ALGEBRAIC[(200 * offset) + h5_i]*ALGEBRAIC[(200 * offset) + h2_i]*CONSTANTS[(208 * offset) + wna];
ALGEBRAIC[(200 * offset) + k8_i] =  ALGEBRAIC[(200 * offset) + h8_i]*CONSTANTS[(208 * offset) + h11_i]*CONSTANTS[(208 * offset) + wna];
ALGEBRAIC[(200 * offset) + h9_i] = 1.00000/ALGEBRAIC[(200 * offset) + h7_i];
ALGEBRAIC[(200 * offset) + k3p_i] =  ALGEBRAIC[(200 * offset) + h9_i]*CONSTANTS[(208 * offset) + wca];
ALGEBRAIC[(200 * offset) + k3_i] = ALGEBRAIC[(200 * offset) + k3p_i]+ALGEBRAIC[(200 * offset) + k3pp_i];
ALGEBRAIC[(200 * offset) + hca] = exp(( CONSTANTS[(208 * offset) + qca]*STATES[(49 * offset) + V]*CONSTANTS[(208 * offset) + F])/( CONSTANTS[(208 * offset) + R]*CONSTANTS[(208 * offset) + T]));
ALGEBRAIC[(200 * offset) + h3_i] = 1.00000/ALGEBRAIC[(200 * offset) + h1_i];
ALGEBRAIC[(200 * offset) + k4p_i] = ( ALGEBRAIC[(200 * offset) + h3_i]*CONSTANTS[(208 * offset) + wca])/ALGEBRAIC[(200 * offset) + hca];
ALGEBRAIC[(200 * offset) + k4_i] = ALGEBRAIC[(200 * offset) + k4p_i]+ALGEBRAIC[(200 * offset) + k4pp_i];
ALGEBRAIC[(200 * offset) + h6_i] = 1.00000/ALGEBRAIC[(200 * offset) + h4_i];
ALGEBRAIC[(200 * offset) + k6_i] =  ALGEBRAIC[(200 * offset) + h6_i]*STATES[(49 * offset) + cai]*CONSTANTS[(208 * offset) + kcaon];
ALGEBRAIC[(200 * offset) + x1_i] =  CONSTANTS[(208 * offset) + k2_i]*ALGEBRAIC[(200 * offset) + k4_i]*(ALGEBRAIC[(200 * offset) + k7_i]+ALGEBRAIC[(200 * offset) + k6_i])+ CONSTANTS[(208 * offset) + k5_i]*ALGEBRAIC[(200 * offset) + k7_i]*(CONSTANTS[(208 * offset) + k2_i]+ALGEBRAIC[(200 * offset) + k3_i]);
ALGEBRAIC[(200 * offset) + x2_i] =  CONSTANTS[(208 * offset) + k1_i]*ALGEBRAIC[(200 * offset) + k7_i]*(ALGEBRAIC[(200 * offset) + k4_i]+CONSTANTS[(208 * offset) + k5_i])+ ALGEBRAIC[(200 * offset) + k4_i]*ALGEBRAIC[(200 * offset) + k6_i]*(CONSTANTS[(208 * offset) + k1_i]+ALGEBRAIC[(200 * offset) + k8_i]);
ALGEBRAIC[(200 * offset) + x3_i] =  CONSTANTS[(208 * offset) + k1_i]*ALGEBRAIC[(200 * offset) + k3_i]*(ALGEBRAIC[(200 * offset) + k7_i]+ALGEBRAIC[(200 * offset) + k6_i])+ ALGEBRAIC[(200 * offset) + k8_i]*ALGEBRAIC[(200 * offset) + k6_i]*(CONSTANTS[(208 * offset) + k2_i]+ALGEBRAIC[(200 * offset) + k3_i]);
ALGEBRAIC[(200 * offset) + x4_i] =  CONSTANTS[(208 * offset) + k2_i]*ALGEBRAIC[(200 * offset) + k8_i]*(ALGEBRAIC[(200 * offset) + k4_i]+CONSTANTS[(208 * offset) + k5_i])+ ALGEBRAIC[(200 * offset) + k3_i]*CONSTANTS[(208 * offset) + k5_i]*(CONSTANTS[(208 * offset) + k1_i]+ALGEBRAIC[(200 * offset) + k8_i]);
ALGEBRAIC[(200 * offset) + E1_i] = ALGEBRAIC[(200 * offset) + x1_i]/(ALGEBRAIC[(200 * offset) + x1_i]+ALGEBRAIC[(200 * offset) + x2_i]+ALGEBRAIC[(200 * offset) + x3_i]+ALGEBRAIC[(200 * offset) + x4_i]);
ALGEBRAIC[(200 * offset) + E2_i] = ALGEBRAIC[(200 * offset) + x2_i]/(ALGEBRAIC[(200 * offset) + x1_i]+ALGEBRAIC[(200 * offset) + x2_i]+ALGEBRAIC[(200 * offset) + x3_i]+ALGEBRAIC[(200 * offset) + x4_i]);
ALGEBRAIC[(200 * offset) + E3_i] = ALGEBRAIC[(200 * offset) + x3_i]/(ALGEBRAIC[(200 * offset) + x1_i]+ALGEBRAIC[(200 * offset) + x2_i]+ALGEBRAIC[(200 * offset) + x3_i]+ALGEBRAIC[(200 * offset) + x4_i]);
ALGEBRAIC[(200 * offset) + E4_i] = ALGEBRAIC[(200 * offset) + x4_i]/(ALGEBRAIC[(200 * offset) + x1_i]+ALGEBRAIC[(200 * offset) + x2_i]+ALGEBRAIC[(200 * offset) + x3_i]+ALGEBRAIC[(200 * offset) + x4_i]);
ALGEBRAIC[(200 * offset) + JncxNa_i] = ( 3.00000*( ALGEBRAIC[(200 * offset) + E4_i]*ALGEBRAIC[(200 * offset) + k7_i] -  ALGEBRAIC[(200 * offset) + E1_i]*ALGEBRAIC[(200 * offset) + k8_i])+ ALGEBRAIC[(200 * offset) + E3_i]*ALGEBRAIC[(200 * offset) + k4pp_i]) -  ALGEBRAIC[(200 * offset) + E2_i]*ALGEBRAIC[(200 * offset) + k3pp_i];
ALGEBRAIC[(200 * offset) + JncxCa_i] =  ALGEBRAIC[(200 * offset) + E2_i]*CONSTANTS[(208 * offset) + k2_i] -  ALGEBRAIC[(200 * offset) + E1_i]*CONSTANTS[(208 * offset) + k1_i];
ALGEBRAIC[(200 * offset) + INaCa_i] =  0.800000*CONSTANTS[(208 * offset) + Gncx]*ALGEBRAIC[(200 * offset) + allo_i]*( CONSTANTS[(208 * offset) + zna]*ALGEBRAIC[(200 * offset) + JncxNa_i]+ CONSTANTS[(208 * offset) + zca]*ALGEBRAIC[(200 * offset) + JncxCa_i]);
ALGEBRAIC[(200 * offset) + A_Nab] = ( CONSTANTS[(208 * offset) + PNab]*CONSTANTS[(208 * offset) + ffrt]*( STATES[(49 * offset) + nai]*exp(ALGEBRAIC[(200 * offset) + vfrt]) - CONSTANTS[(208 * offset) + nao]))/CONSTANTS[(208 * offset) + B_Nab];
ALGEBRAIC[(200 * offset) + U_Nab] =  CONSTANTS[(208 * offset) + B_Nab]*(STATES[(49 * offset) + V] - CONSTANTS[(208 * offset) + v0_Nab]);
ALGEBRAIC[(200 * offset) + INab] = (- 1.00000e-07<=ALGEBRAIC[(200 * offset) + U_Nab]&&ALGEBRAIC[(200 * offset) + U_Nab]<=1.00000e-07 ?  ALGEBRAIC[(200 * offset) + A_Nab]*(1.00000 -  0.500000*ALGEBRAIC[(200 * offset) + U_Nab]) : ( ALGEBRAIC[(200 * offset) + A_Nab]*ALGEBRAIC[(200 * offset) + U_Nab])/(exp(ALGEBRAIC[(200 * offset) + U_Nab]) - 1.00000));
ALGEBRAIC[(200 * offset) + JdiffNa] = (STATES[(49 * offset) + nass] - STATES[(49 * offset) + nai])/2.00000;
ALGEBRAIC[(200 * offset) + A_2] = ( 0.750000*CONSTANTS[(208 * offset) + ffrt]*( STATES[(49 * offset) + nass]*exp(ALGEBRAIC[(200 * offset) + vfrt]) - CONSTANTS[(208 * offset) + nao]))/CONSTANTS[(208 * offset) + B_2];
ALGEBRAIC[(200 * offset) + U_2] =  CONSTANTS[(208 * offset) + B_2]*(STATES[(49 * offset) + V] - CONSTANTS[(208 * offset) + v0_CaL]);
ALGEBRAIC[(200 * offset) + PhiCaNa] = (- 1.00000e-07<=ALGEBRAIC[(200 * offset) + U_2]&&ALGEBRAIC[(200 * offset) + U_2]<=1.00000e-07 ?  ALGEBRAIC[(200 * offset) + A_2]*(1.00000 -  0.500000*ALGEBRAIC[(200 * offset) + U_2]) : ( ALGEBRAIC[(200 * offset) + A_2]*ALGEBRAIC[(200 * offset) + U_2])/(exp(ALGEBRAIC[(200 * offset) + U_2]) - 1.00000));
ALGEBRAIC[(200 * offset) + ICaNa] =  (1.00000 - ALGEBRAIC[(200 * offset) + fICaLp])*CONSTANTS[(208 * offset) + PCaNa]*ALGEBRAIC[(200 * offset) + PhiCaNa]*STATES[(49 * offset) + d]*( ALGEBRAIC[(200 * offset) + f]*(1.00000 - STATES[(49 * offset) + nca])+ STATES[(49 * offset) + jca]*ALGEBRAIC[(200 * offset) + fca]*STATES[(49 * offset) + nca])+ ALGEBRAIC[(200 * offset) + fICaLp]*CONSTANTS[(208 * offset) + PCaNap]*ALGEBRAIC[(200 * offset) + PhiCaNa]*STATES[(49 * offset) + d]*( ALGEBRAIC[(200 * offset) + fp]*(1.00000 - STATES[(49 * offset) + nca])+ STATES[(49 * offset) + jca]*ALGEBRAIC[(200 * offset) + fcap]*STATES[(49 * offset) + nca]);
ALGEBRAIC[(200 * offset) + allo_ss] = 1.00000/(1.00000+pow(CONSTANTS[(208 * offset) + KmCaAct]/STATES[(49 * offset) + cass], 2.00000));
ALGEBRAIC[(200 * offset) + h7_ss] = 1.00000+ (CONSTANTS[(208 * offset) + nao]/CONSTANTS[(208 * offset) + kna3])*(1.00000+1.00000/ALGEBRAIC[(200 * offset) + hna]);
ALGEBRAIC[(200 * offset) + h8_ss] = CONSTANTS[(208 * offset) + nao]/( CONSTANTS[(208 * offset) + kna3]*ALGEBRAIC[(200 * offset) + hna]*ALGEBRAIC[(200 * offset) + h7_ss]);
ALGEBRAIC[(200 * offset) + k3pp_ss] =  ALGEBRAIC[(200 * offset) + h8_ss]*CONSTANTS[(208 * offset) + wnaca];
ALGEBRAIC[(200 * offset) + h1_ss] = 1.00000+ (STATES[(49 * offset) + nass]/CONSTANTS[(208 * offset) + kna3])*(1.00000+ALGEBRAIC[(200 * offset) + hna]);
ALGEBRAIC[(200 * offset) + h2_ss] = ( STATES[(49 * offset) + nass]*ALGEBRAIC[(200 * offset) + hna])/( CONSTANTS[(208 * offset) + kna3]*ALGEBRAIC[(200 * offset) + h1_ss]);
ALGEBRAIC[(200 * offset) + k4pp_ss] =  ALGEBRAIC[(200 * offset) + h2_ss]*CONSTANTS[(208 * offset) + wnaca];
ALGEBRAIC[(200 * offset) + h4_ss] = 1.00000+ (STATES[(49 * offset) + nass]/CONSTANTS[(208 * offset) + kna1])*(1.00000+STATES[(49 * offset) + nass]/CONSTANTS[(208 * offset) + kna2]);
ALGEBRAIC[(200 * offset) + h5_ss] = ( STATES[(49 * offset) + nass]*STATES[(49 * offset) + nass])/( ALGEBRAIC[(200 * offset) + h4_ss]*CONSTANTS[(208 * offset) + kna1]*CONSTANTS[(208 * offset) + kna2]);
ALGEBRAIC[(200 * offset) + k7_ss] =  ALGEBRAIC[(200 * offset) + h5_ss]*ALGEBRAIC[(200 * offset) + h2_ss]*CONSTANTS[(208 * offset) + wna];
ALGEBRAIC[(200 * offset) + k8_ss] =  ALGEBRAIC[(200 * offset) + h8_ss]*CONSTANTS[(208 * offset) + h11_ss]*CONSTANTS[(208 * offset) + wna];
ALGEBRAIC[(200 * offset) + h9_ss] = 1.00000/ALGEBRAIC[(200 * offset) + h7_ss];
ALGEBRAIC[(200 * offset) + k3p_ss] =  ALGEBRAIC[(200 * offset) + h9_ss]*CONSTANTS[(208 * offset) + wca];
ALGEBRAIC[(200 * offset) + k3_ss] = ALGEBRAIC[(200 * offset) + k3p_ss]+ALGEBRAIC[(200 * offset) + k3pp_ss];
ALGEBRAIC[(200 * offset) + h3_ss] = 1.00000/ALGEBRAIC[(200 * offset) + h1_ss];
ALGEBRAIC[(200 * offset) + k4p_ss] = ( ALGEBRAIC[(200 * offset) + h3_ss]*CONSTANTS[(208 * offset) + wca])/ALGEBRAIC[(200 * offset) + hca];
ALGEBRAIC[(200 * offset) + k4_ss] = ALGEBRAIC[(200 * offset) + k4p_ss]+ALGEBRAIC[(200 * offset) + k4pp_ss];
ALGEBRAIC[(200 * offset) + h6_ss] = 1.00000/ALGEBRAIC[(200 * offset) + h4_ss];
ALGEBRAIC[(200 * offset) + k6_ss] =  ALGEBRAIC[(200 * offset) + h6_ss]*STATES[(49 * offset) + cass]*CONSTANTS[(208 * offset) + kcaon];
ALGEBRAIC[(200 * offset) + x1_ss] =  CONSTANTS[(208 * offset) + k2_ss]*ALGEBRAIC[(200 * offset) + k4_ss]*(ALGEBRAIC[(200 * offset) + k7_ss]+ALGEBRAIC[(200 * offset) + k6_ss])+ CONSTANTS[(208 * offset) + k5_ss]*ALGEBRAIC[(200 * offset) + k7_ss]*(CONSTANTS[(208 * offset) + k2_ss]+ALGEBRAIC[(200 * offset) + k3_ss]);
ALGEBRAIC[(200 * offset) + x2_ss] =  CONSTANTS[(208 * offset) + k1_ss]*ALGEBRAIC[(200 * offset) + k7_ss]*(ALGEBRAIC[(200 * offset) + k4_ss]+CONSTANTS[(208 * offset) + k5_ss])+ ALGEBRAIC[(200 * offset) + k4_ss]*ALGEBRAIC[(200 * offset) + k6_ss]*(CONSTANTS[(208 * offset) + k1_ss]+ALGEBRAIC[(200 * offset) + k8_ss]);
ALGEBRAIC[(200 * offset) + x3_ss] =  CONSTANTS[(208 * offset) + k1_ss]*ALGEBRAIC[(200 * offset) + k3_ss]*(ALGEBRAIC[(200 * offset) + k7_ss]+ALGEBRAIC[(200 * offset) + k6_ss])+ ALGEBRAIC[(200 * offset) + k8_ss]*ALGEBRAIC[(200 * offset) + k6_ss]*(CONSTANTS[(208 * offset) + k2_ss]+ALGEBRAIC[(200 * offset) + k3_ss]);
ALGEBRAIC[(200 * offset) + x4_ss] =  CONSTANTS[(208 * offset) + k2_ss]*ALGEBRAIC[(200 * offset) + k8_ss]*(ALGEBRAIC[(200 * offset) + k4_ss]+CONSTANTS[(208 * offset) + k5_ss])+ ALGEBRAIC[(200 * offset) + k3_ss]*CONSTANTS[(208 * offset) + k5_ss]*(CONSTANTS[(208 * offset) + k1_ss]+ALGEBRAIC[(200 * offset) + k8_ss]);
ALGEBRAIC[(200 * offset) + E1_ss] = ALGEBRAIC[(200 * offset) + x1_ss]/(ALGEBRAIC[(200 * offset) + x1_ss]+ALGEBRAIC[(200 * offset) + x2_ss]+ALGEBRAIC[(200 * offset) + x3_ss]+ALGEBRAIC[(200 * offset) + x4_ss]);
ALGEBRAIC[(200 * offset) + E2_ss] = ALGEBRAIC[(200 * offset) + x2_ss]/(ALGEBRAIC[(200 * offset) + x1_ss]+ALGEBRAIC[(200 * offset) + x2_ss]+ALGEBRAIC[(200 * offset) + x3_ss]+ALGEBRAIC[(200 * offset) + x4_ss]);
ALGEBRAIC[(200 * offset) + E3_ss] = ALGEBRAIC[(200 * offset) + x3_ss]/(ALGEBRAIC[(200 * offset) + x1_ss]+ALGEBRAIC[(200 * offset) + x2_ss]+ALGEBRAIC[(200 * offset) + x3_ss]+ALGEBRAIC[(200 * offset) + x4_ss]);
ALGEBRAIC[(200 * offset) + E4_ss] = ALGEBRAIC[(200 * offset) + x4_ss]/(ALGEBRAIC[(200 * offset) + x1_ss]+ALGEBRAIC[(200 * offset) + x2_ss]+ALGEBRAIC[(200 * offset) + x3_ss]+ALGEBRAIC[(200 * offset) + x4_ss]);
ALGEBRAIC[(200 * offset) + JncxNa_ss] = ( 3.00000*( ALGEBRAIC[(200 * offset) + E4_ss]*ALGEBRAIC[(200 * offset) + k7_ss] -  ALGEBRAIC[(200 * offset) + E1_ss]*ALGEBRAIC[(200 * offset) + k8_ss])+ ALGEBRAIC[(200 * offset) + E3_ss]*ALGEBRAIC[(200 * offset) + k4pp_ss]) -  ALGEBRAIC[(200 * offset) + E2_ss]*ALGEBRAIC[(200 * offset) + k3pp_ss];
ALGEBRAIC[(200 * offset) + JncxCa_ss] =  ALGEBRAIC[(200 * offset) + E2_ss]*CONSTANTS[(208 * offset) + k2_ss] -  ALGEBRAIC[(200 * offset) + E1_ss]*CONSTANTS[(208 * offset) + k1_ss];
ALGEBRAIC[(200 * offset) + INaCa_ss] =  0.200000*CONSTANTS[(208 * offset) + Gncx]*ALGEBRAIC[(200 * offset) + allo_ss]*( CONSTANTS[(208 * offset) + zna]*ALGEBRAIC[(200 * offset) + JncxNa_ss]+ CONSTANTS[(208 * offset) + zca]*ALGEBRAIC[(200 * offset) + JncxCa_ss]);
ALGEBRAIC[(200 * offset) + IpCa] = ( CONSTANTS[(208 * offset) + GpCa]*STATES[(49 * offset) + cai])/(CONSTANTS[(208 * offset) + KmCap]+STATES[(49 * offset) + cai]);
ALGEBRAIC[(200 * offset) + A_Cab] = ( CONSTANTS[(208 * offset) + PCab]*4.00000*CONSTANTS[(208 * offset) + ffrt]*( STATES[(49 * offset) + cai]*exp( 2.00000*ALGEBRAIC[(200 * offset) + vfrt]) -  0.341000*CONSTANTS[(208 * offset) + cao]))/CONSTANTS[(208 * offset) + B_Cab];
ALGEBRAIC[(200 * offset) + U_Cab] =  CONSTANTS[(208 * offset) + B_Cab]*(STATES[(49 * offset) + V] - CONSTANTS[(208 * offset) + v0_Cab]);
ALGEBRAIC[(200 * offset) + ICab] = (- 1.00000e-07<=ALGEBRAIC[(200 * offset) + U_Cab]&&ALGEBRAIC[(200 * offset) + U_Cab]<=1.00000e-07 ?  ALGEBRAIC[(200 * offset) + A_Cab]*(1.00000 -  0.500000*ALGEBRAIC[(200 * offset) + U_Cab]) : ( ALGEBRAIC[(200 * offset) + A_Cab]*ALGEBRAIC[(200 * offset) + U_Cab])/(exp(ALGEBRAIC[(200 * offset) + U_Cab]) - 1.00000));
ALGEBRAIC[(200 * offset) + Jdiff] = (STATES[(49 * offset) + cass] - STATES[(49 * offset) + cai])/0.200000;
ALGEBRAIC[(200 * offset) + fJrelp] = 1.00000/(1.00000+CONSTANTS[(208 * offset) + KmCaMK]/ALGEBRAIC[(200 * offset) + CaMKa]);
ALGEBRAIC[(200 * offset) + Jrel] =  CONSTANTS[(208 * offset) + Jrel_scaling_factor]*( (1.00000 - ALGEBRAIC[(200 * offset) + fJrelp])*STATES[(49 * offset) + Jrelnp]+ ALGEBRAIC[(200 * offset) + fJrelp]*STATES[(49 * offset) + Jrelp]);
ALGEBRAIC[(200 * offset) + Bcass] = 1.00000/(1.00000+( CONSTANTS[(208 * offset) + BSRmax]*CONSTANTS[(208 * offset) + KmBSR])/pow(CONSTANTS[(208 * offset) + KmBSR]+STATES[(49 * offset) + cass], 2.00000)+( CONSTANTS[(208 * offset) + BSLmax]*CONSTANTS[(208 * offset) + KmBSL])/pow(CONSTANTS[(208 * offset) + KmBSL]+STATES[(49 * offset) + cass], 2.00000));
ALGEBRAIC[(200 * offset) + Jupnp] = ( CONSTANTS[(208 * offset) + upScale]*0.00437500*STATES[(49 * offset) + cai])/(STATES[(49 * offset) + cai]+0.000920000);
ALGEBRAIC[(200 * offset) + Jupp] = ( CONSTANTS[(208 * offset) + upScale]*2.75000*0.00437500*STATES[(49 * offset) + cai])/((STATES[(49 * offset) + cai]+0.000920000) - 0.000170000);
ALGEBRAIC[(200 * offset) + fJupp] = 1.00000/(1.00000+CONSTANTS[(208 * offset) + KmCaMK]/ALGEBRAIC[(200 * offset) + CaMKa]);
//cvar addition
ALGEBRAIC[(200 * offset) + Jleak] = CONSTANTS[(208 * offset) + Jleak_b] * ( 0.00393750*STATES[(49 * offset) + cansr])/15.0000;
ALGEBRAIC[(200 * offset) + Jup] =  CONSTANTS[(208 * offset) + Jup_b]*(( (1.00000 - ALGEBRAIC[(200 * offset) + fJupp])*ALGEBRAIC[(200 * offset) + Jupnp]+ ALGEBRAIC[(200 * offset) + fJupp]*ALGEBRAIC[(200 * offset) + Jupp]) - ALGEBRAIC[(200 * offset) + Jleak]);
ALGEBRAIC[(200 * offset) + Bcai] = 1.00000/(1.00000+( CONSTANTS[(208 * offset) + cmdnmax]*CONSTANTS[(208 * offset) + kmcmdn])/pow(CONSTANTS[(208 * offset) + kmcmdn]+STATES[(49 * offset) + cai], 2.00000)+( CONSTANTS[(208 * offset) + trpnmax]*CONSTANTS[(208 * offset) + kmtrpn])/pow(CONSTANTS[(208 * offset) + kmtrpn]+STATES[(49 * offset) + cai], 2.00000));
ALGEBRAIC[(200 * offset) + Jtr] = CONSTANTS[(208 * offset) + Jtr_b] * (STATES[(49 * offset) + cansr] - STATES[(49 * offset) + cajsr])/100.000;
ALGEBRAIC[(200 * offset) + Bcajsr] = 1.00000/(1.00000+( CONSTANTS[(208 * offset) + csqnmax]*CONSTANTS[(208 * offset) + kmcsqn])/pow(CONSTANTS[(208 * offset) + kmcsqn]+STATES[(49 * offset) + cajsr], 2.00000));

//RATES[(49 * offset) + D] = CONSTANTS[(208 * offset) + cnc];
RATES[(49 * offset) + D] = 0.;
RATES[(49 * offset) + IC1] = (- ( CONSTANTS[(208 * offset) + A11]*exp( CONSTANTS[(208 * offset) + B11]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC1]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q11]))/10.0000) -  CONSTANTS[(208 * offset) + A21]*exp( CONSTANTS[(208 * offset) + B21]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q21]))/10.0000))+ CONSTANTS[(208 * offset) + A51]*exp( CONSTANTS[(208 * offset) + B51]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C1]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q51]))/10.0000)) -  CONSTANTS[(208 * offset) + A61]*exp( CONSTANTS[(208 * offset) + B61]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC1]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q61]))/10.0000);
RATES[(49 * offset) + IC2] = ((( CONSTANTS[(208 * offset) + A11]*exp( CONSTANTS[(208 * offset) + B11]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC1]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q11]))/10.0000) -  CONSTANTS[(208 * offset) + A21]*exp( CONSTANTS[(208 * offset) + B21]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q21]))/10.0000)) - ( CONSTANTS[(208 * offset) + A3]*exp( CONSTANTS[(208 * offset) + B3]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q3]))/10.0000) -  CONSTANTS[(208 * offset) + A4]*exp( CONSTANTS[(208 * offset) + B4]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IO]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q4]))/10.0000)))+ CONSTANTS[(208 * offset) + A52]*exp( CONSTANTS[(208 * offset) + B52]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q52]))/10.0000)) -  CONSTANTS[(208 * offset) + A62]*exp( CONSTANTS[(208 * offset) + B62]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q62]))/10.0000);
RATES[(49 * offset) + C1] = - ( CONSTANTS[(208 * offset) + A1]*exp( CONSTANTS[(208 * offset) + B1]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C1]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q1]))/10.0000) -  CONSTANTS[(208 * offset) + A2]*exp( CONSTANTS[(208 * offset) + B2]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q2]))/10.0000)) - ( CONSTANTS[(208 * offset) + A51]*exp( CONSTANTS[(208 * offset) + B51]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C1]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q51]))/10.0000) -  CONSTANTS[(208 * offset) + A61]*exp( CONSTANTS[(208 * offset) + B61]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC1]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q61]))/10.0000));
RATES[(49 * offset) + C2] = (( CONSTANTS[(208 * offset) + A1]*exp( CONSTANTS[(208 * offset) + B1]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C1]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q1]))/10.0000) -  CONSTANTS[(208 * offset) + A2]*exp( CONSTANTS[(208 * offset) + B2]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q2]))/10.0000)) - ( CONSTANTS[(208 * offset) + A31]*exp( CONSTANTS[(208 * offset) + B31]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q31]))/10.0000) -  CONSTANTS[(208 * offset) + A41]*exp( CONSTANTS[(208 * offset) + B41]*STATES[(49 * offset) + V])*STATES[(49 * offset) + O]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q41]))/10.0000))) - ( CONSTANTS[(208 * offset) + A52]*exp( CONSTANTS[(208 * offset) + B52]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q52]))/10.0000) -  CONSTANTS[(208 * offset) + A62]*exp( CONSTANTS[(208 * offset) + B62]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q62]))/10.0000));
RATES[(49 * offset) + O] = (( CONSTANTS[(208 * offset) + A31]*exp( CONSTANTS[(208 * offset) + B31]*STATES[(49 * offset) + V])*STATES[(49 * offset) + C2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q31]))/10.0000) -  CONSTANTS[(208 * offset) + A41]*exp( CONSTANTS[(208 * offset) + B41]*STATES[(49 * offset) + V])*STATES[(49 * offset) + O]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q41]))/10.0000)) - ( CONSTANTS[(208 * offset) + A53]*exp( CONSTANTS[(208 * offset) + B53]*STATES[(49 * offset) + V])*STATES[(49 * offset) + O]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q53]))/10.0000) -  CONSTANTS[(208 * offset) + A63]*exp( CONSTANTS[(208 * offset) + B63]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IO]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q63]))/10.0000))) - ( (( CONSTANTS[(208 * offset) + Kmax]*CONSTANTS[(208 * offset) + Ku]*pow( STATES[(49 * offset) + D],CONSTANTS[(208 * offset) + n]))/(pow( STATES[(49 * offset) + D],CONSTANTS[(208 * offset) + n])+CONSTANTS[(208 * offset) + halfmax]))*STATES[(49 * offset) + O] -  CONSTANTS[(208 * offset) + Ku]*STATES[(49 * offset) + Obound]);
RATES[(49 * offset) + IO] = ((( CONSTANTS[(208 * offset) + A3]*exp( CONSTANTS[(208 * offset) + B3]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IC2]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q3]))/10.0000) -  CONSTANTS[(208 * offset) + A4]*exp( CONSTANTS[(208 * offset) + B4]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IO]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q4]))/10.0000))+ CONSTANTS[(208 * offset) + A53]*exp( CONSTANTS[(208 * offset) + B53]*STATES[(49 * offset) + V])*STATES[(49 * offset) + O]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q53]))/10.0000)) -  CONSTANTS[(208 * offset) + A63]*exp( CONSTANTS[(208 * offset) + B63]*STATES[(49 * offset) + V])*STATES[(49 * offset) + IO]*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q63]))/10.0000)) - ( (( CONSTANTS[(208 * offset) + Kmax]*CONSTANTS[(208 * offset) + Ku]*pow( STATES[(49 * offset) + D],CONSTANTS[(208 * offset) + n]))/(pow( STATES[(49 * offset) + D],CONSTANTS[(208 * offset) + n])+CONSTANTS[(208 * offset) + halfmax]))*STATES[(49 * offset) + IO] -  (( CONSTANTS[(208 * offset) + Ku]*CONSTANTS[(208 * offset) + A53]*exp( CONSTANTS[(208 * offset) + B53]*STATES[(49 * offset) + V])*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q53]))/10.0000))/( CONSTANTS[(208 * offset) + A63]*exp( CONSTANTS[(208 * offset) + B63]*STATES[(49 * offset) + V])*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q63]))/10.0000)))*STATES[(49 * offset) + IObound]);
RATES[(49 * offset) + IObound] = (( (( CONSTANTS[(208 * offset) + Kmax]*CONSTANTS[(208 * offset) + Ku]*pow( STATES[(49 * offset) + D],CONSTANTS[(208 * offset) + n]))/(pow( STATES[(49 * offset) + D],CONSTANTS[(208 * offset) + n])+CONSTANTS[(208 * offset) + halfmax]))*STATES[(49 * offset) + IO] -  (( CONSTANTS[(208 * offset) + Ku]*CONSTANTS[(208 * offset) + A53]*exp( CONSTANTS[(208 * offset) + B53]*STATES[(49 * offset) + V])*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q53]))/10.0000))/( CONSTANTS[(208 * offset) + A63]*exp( CONSTANTS[(208 * offset) + B63]*STATES[(49 * offset) + V])*exp(( (CONSTANTS[(208 * offset) + Temp] - 20.0000)*log(CONSTANTS[(208 * offset) + q63]))/10.0000)))*STATES[(49 * offset) + IObound])+ (CONSTANTS[(208 * offset) + Kt]/(1.00000+exp(- (STATES[(49 * offset) + V] - CONSTANTS[(208 * offset) + Vhalf])/6.78900)))*STATES[(49 * offset) + Cbound]) -  CONSTANTS[(208 * offset) + Kt]*STATES[(49 * offset) + IObound];
RATES[(49 * offset) + Obound] = (( (( CONSTANTS[(208 * offset) + Kmax]*CONSTANTS[(208 * offset) + Ku]*pow( STATES[(49 * offset) + D],CONSTANTS[(208 * offset) + n]))/(pow( STATES[(49 * offset) + D],CONSTANTS[(208 * offset) + n])+CONSTANTS[(208 * offset) + halfmax]))*STATES[(49 * offset) + O] -  CONSTANTS[(208 * offset) + Ku]*STATES[(49 * offset) + Obound])+ (CONSTANTS[(208 * offset) + Kt]/(1.00000+exp(- (STATES[(49 * offset) + V] - CONSTANTS[(208 * offset) + Vhalf])/6.78900)))*STATES[(49 * offset) + Cbound]) -  CONSTANTS[(208 * offset) + Kt]*STATES[(49 * offset) + Obound];
RATES[(49 * offset) + Cbound] = - ( (CONSTANTS[(208 * offset) + Kt]/(1.00000+exp(- (STATES[(49 * offset) + V] - CONSTANTS[(208 * offset) + Vhalf])/6.78900)))*STATES[(49 * offset) + Cbound] -  CONSTANTS[(208 * offset) + Kt]*STATES[(49 * offset) + Obound]) - ( (CONSTANTS[(208 * offset) + Kt]/(1.00000+exp(- (STATES[(49 * offset) + V] - CONSTANTS[(208 * offset) + Vhalf])/6.78900)))*STATES[(49 * offset) + Cbound] -  CONSTANTS[(208 * offset) + Kt]*STATES[(49 * offset) + IObound]);
RATES[(49 * offset) + hL] = (ALGEBRAIC[(200 * offset) + hLss] - STATES[(49 * offset) + hL])/CONSTANTS[(208 * offset) + thL];
RATES[(49 * offset) + hLp] = (ALGEBRAIC[(200 * offset) + hLssp] - STATES[(49 * offset) + hLp])/CONSTANTS[(208 * offset) + thLp];
RATES[(49 * offset) + m] = (ALGEBRAIC[(200 * offset) + mss] - STATES[(49 * offset) + m])/ALGEBRAIC[(200 * offset) + tm];
RATES[(49 * offset) + hf] = (ALGEBRAIC[(200 * offset) + hss] - STATES[(49 * offset) + hf])/ALGEBRAIC[(200 * offset) + thf];
RATES[(49 * offset) + hs] = (ALGEBRAIC[(200 * offset) + hss] - STATES[(49 * offset) + hs])/ALGEBRAIC[(200 * offset) + ths];
RATES[(49 * offset) + a] = (ALGEBRAIC[(200 * offset) + ass] - STATES[(49 * offset) + a])/ALGEBRAIC[(200 * offset) + ta];
RATES[(49 * offset) + d] = (ALGEBRAIC[(200 * offset) + dss] - STATES[(49 * offset) + d])/ALGEBRAIC[(200 * offset) + td];
RATES[(49 * offset) + ff] = (ALGEBRAIC[(200 * offset) + fss] - STATES[(49 * offset) + ff])/ALGEBRAIC[(200 * offset) + tff];
RATES[(49 * offset) + fs] = (ALGEBRAIC[(200 * offset) + fss] - STATES[(49 * offset) + fs])/ALGEBRAIC[(200 * offset) + tfs];
RATES[(49 * offset) + jca] = (ALGEBRAIC[(200 * offset) + fcass] - STATES[(49 * offset) + jca])/CONSTANTS[(208 * offset) + tjca];
RATES[(49 * offset) + nca] =  ALGEBRAIC[(200 * offset) + anca]*CONSTANTS[(208 * offset) + k2n] -  STATES[(49 * offset) + nca]*ALGEBRAIC[(200 * offset) + km2n];
RATES[(49 * offset) + xs1] = (ALGEBRAIC[(200 * offset) + xs1ss] - STATES[(49 * offset) + xs1])/ALGEBRAIC[(200 * offset) + txs1];
RATES[(49 * offset) + xk1] = (ALGEBRAIC[(200 * offset) + xk1ss] - STATES[(49 * offset) + xk1])/ALGEBRAIC[(200 * offset) + txk1];
RATES[(49 * offset) + CaMKt] =  CONSTANTS[(208 * offset) + aCaMK]*ALGEBRAIC[(200 * offset) + CaMKb]*(ALGEBRAIC[(200 * offset) + CaMKb]+STATES[(49 * offset) + CaMKt]) -  CONSTANTS[(208 * offset) + bCaMK]*STATES[(49 * offset) + CaMKt];
RATES[(49 * offset) + j] = (ALGEBRAIC[(200 * offset) + jss] - STATES[(49 * offset) + j])/ALGEBRAIC[(200 * offset) + tj];
RATES[(49 * offset) + ap] = (ALGEBRAIC[(200 * offset) + assp] - STATES[(49 * offset) + ap])/ALGEBRAIC[(200 * offset) + ta];
RATES[(49 * offset) + fcaf] = (ALGEBRAIC[(200 * offset) + fcass] - STATES[(49 * offset) + fcaf])/ALGEBRAIC[(200 * offset) + tfcaf];
RATES[(49 * offset) + fcas] = (ALGEBRAIC[(200 * offset) + fcass] - STATES[(49 * offset) + fcas])/ALGEBRAIC[(200 * offset) + tfcas];
RATES[(49 * offset) + ffp] = (ALGEBRAIC[(200 * offset) + fss] - STATES[(49 * offset) + ffp])/ALGEBRAIC[(200 * offset) + tffp];
RATES[(49 * offset) + xs2] = (ALGEBRAIC[(200 * offset) + xs2ss] - STATES[(49 * offset) + xs2])/ALGEBRAIC[(200 * offset) + txs2];
RATES[(49 * offset) + hsp] = (ALGEBRAIC[(200 * offset) + hssp] - STATES[(49 * offset) + hsp])/ALGEBRAIC[(200 * offset) + thsp];
RATES[(49 * offset) + jp] = (ALGEBRAIC[(200 * offset) + jss] - STATES[(49 * offset) + jp])/ALGEBRAIC[(200 * offset) + tjp];
RATES[(49 * offset) + mL] = (ALGEBRAIC[(200 * offset) + mLss] - STATES[(49 * offset) + mL])/ALGEBRAIC[(200 * offset) + tmL];
RATES[(49 * offset) + fcafp] = (ALGEBRAIC[(200 * offset) + fcass] - STATES[(49 * offset) + fcafp])/ALGEBRAIC[(200 * offset) + tfcafp];
RATES[(49 * offset) + iF] = (ALGEBRAIC[(200 * offset) + iss] - STATES[(49 * offset) + iF])/ALGEBRAIC[(200 * offset) + tiF];
RATES[(49 * offset) + iS] = (ALGEBRAIC[(200 * offset) + iss] - STATES[(49 * offset) + iS])/ALGEBRAIC[(200 * offset) + tiS];
RATES[(49 * offset) + iFp] = (ALGEBRAIC[(200 * offset) + iss] - STATES[(49 * offset) + iFp])/ALGEBRAIC[(200 * offset) + tiFp];
RATES[(49 * offset) + iSp] = (ALGEBRAIC[(200 * offset) + iss] - STATES[(49 * offset) + iSp])/ALGEBRAIC[(200 * offset) + tiSp];
RATES[(49 * offset) + Jrelnp] = (ALGEBRAIC[(200 * offset) + Jrel_inf] - STATES[(49 * offset) + Jrelnp])/ALGEBRAIC[(200 * offset) + tau_rel];
RATES[(49 * offset) + Jrelp] = (ALGEBRAIC[(200 * offset) + Jrel_infp] - STATES[(49 * offset) + Jrelp])/ALGEBRAIC[(200 * offset) + tau_relp];
RATES[(49 * offset) + ki] = ( - ((ALGEBRAIC[(200 * offset) + Ito]+ALGEBRAIC[(200 * offset) + IKr]+ALGEBRAIC[(200 * offset) + IKs]+ALGEBRAIC[(200 * offset) + IK1]+ALGEBRAIC[(200 * offset) + IKb]+ALGEBRAIC[(200 * offset) + Istim]) -  2.00000*ALGEBRAIC[(200 * offset) + INaK])*CONSTANTS[(208 * offset) + cm]*CONSTANTS[(208 * offset) + Acap])/( CONSTANTS[(208 * offset) + F]*CONSTANTS[(208 * offset) + vmyo])+( ALGEBRAIC[(200 * offset) + JdiffK]*CONSTANTS[(208 * offset) + vss])/CONSTANTS[(208 * offset) + vmyo];
RATES[(49 * offset) + kss] = ( - ALGEBRAIC[(200 * offset) + ICaK]*CONSTANTS[(208 * offset) + cm]*CONSTANTS[(208 * offset) + Acap])/( CONSTANTS[(208 * offset) + F]*CONSTANTS[(208 * offset) + vss]) - ALGEBRAIC[(200 * offset) + JdiffK];
RATES[(49 * offset) + nai] = ( - (ALGEBRAIC[(200 * offset) + INa]+ALGEBRAIC[(200 * offset) + INaL]+ 3.00000*ALGEBRAIC[(200 * offset) + INaCa_i]+ 3.00000*ALGEBRAIC[(200 * offset) + INaK]+ALGEBRAIC[(200 * offset) + INab])*CONSTANTS[(208 * offset) + Acap]*CONSTANTS[(208 * offset) + cm])/( CONSTANTS[(208 * offset) + F]*CONSTANTS[(208 * offset) + vmyo])+( ALGEBRAIC[(200 * offset) + JdiffNa]*CONSTANTS[(208 * offset) + vss])/CONSTANTS[(208 * offset) + vmyo];
RATES[(49 * offset) + nass] = ( - (ALGEBRAIC[(200 * offset) + ICaNa]+ 3.00000*ALGEBRAIC[(200 * offset) + INaCa_ss])*CONSTANTS[(208 * offset) + cm]*CONSTANTS[(208 * offset) + Acap])/( CONSTANTS[(208 * offset) + F]*CONSTANTS[(208 * offset) + vss]) - ALGEBRAIC[(200 * offset) + JdiffNa];
RATES[(49 * offset) + V] = - (ALGEBRAIC[(200 * offset) + INa]+ALGEBRAIC[(200 * offset) + INaL]+ALGEBRAIC[(200 * offset) + Ito]+ALGEBRAIC[(200 * offset) + ICaL]+ALGEBRAIC[(200 * offset) + ICaNa]+ALGEBRAIC[(200 * offset) + ICaK]+ALGEBRAIC[(200 * offset) + IKr]+ALGEBRAIC[(200 * offset) + IKs]+ALGEBRAIC[(200 * offset) + IK1]+ALGEBRAIC[(200 * offset) + INaCa_i]+ALGEBRAIC[(200 * offset) + INaCa_ss]+ALGEBRAIC[(200 * offset) + INaK]+ALGEBRAIC[(200 * offset) + INab]+ALGEBRAIC[(200 * offset) + IKb]+ALGEBRAIC[(200 * offset) + IpCa]+ALGEBRAIC[(200 * offset) + ICab]+ALGEBRAIC[(200 * offset) + Istim]);
RATES[(49 * offset) + cass] =  ALGEBRAIC[(200 * offset) + Bcass]*((( - (ALGEBRAIC[(200 * offset) + ICaL] -  2.00000*ALGEBRAIC[(200 * offset) + INaCa_ss])*CONSTANTS[(208 * offset) + cm]*CONSTANTS[(208 * offset) + Acap])/( 2.00000*CONSTANTS[(208 * offset) + F]*CONSTANTS[(208 * offset) + vss])+( ALGEBRAIC[(200 * offset) + Jrel]*CONSTANTS[(208 * offset) + vjsr])/CONSTANTS[(208 * offset) + vss]) - ALGEBRAIC[(200 * offset) + Jdiff]);
RATES[(49 * offset) + cai] =  ALGEBRAIC[(200 * offset) + Bcai]*((( - ((ALGEBRAIC[(200 * offset) + IpCa]+ALGEBRAIC[(200 * offset) + ICab]) -  2.00000*ALGEBRAIC[(200 * offset) + INaCa_i])*CONSTANTS[(208 * offset) + cm]*CONSTANTS[(208 * offset) + Acap])/( 2.00000*CONSTANTS[(208 * offset) + F]*CONSTANTS[(208 * offset) + vmyo]) - ( ALGEBRAIC[(200 * offset) + Jup]*CONSTANTS[(208 * offset) + vnsr])/CONSTANTS[(208 * offset) + vmyo])+( ALGEBRAIC[(200 * offset) + Jdiff]*CONSTANTS[(208 * offset) + vss])/CONSTANTS[(208 * offset) + vmyo]);
RATES[(49 * offset) + cansr] = ALGEBRAIC[(200 * offset) + Jup] - ( ALGEBRAIC[(200 * offset) + Jtr]*CONSTANTS[(208 * offset) + vjsr])/CONSTANTS[(208 * offset) + vnsr];
RATES[(49 * offset) + cajsr] =  ALGEBRAIC[(200 * offset) + Bcajsr]*(ALGEBRAIC[(200 * offset) + Jtr] - ALGEBRAIC[(200 * offset) + Jrel]);
}


__device__ void solveAnalytical(double *CONSTANTS, double *STATES, double *ALGEBRAIC, double *RATES, double dt, int offset)
{

}

__device__ void solveEuler( double *STATES, double *RATES, double dt, int offset)
{
    for(int i=0;i<49;i++){
    STATES[(49 * offset) + i] = STATES[(49 * offset) + i] + RATES[(49 * offset) + i] * dt;
    }
}

// double ohara_rudy_cipa_v1_2017::set_time_step(double TIME,
//                                               double time_point,
//                                               double min_time_step,
//                                               double max_time_step,
//                                               double min_dV,
//                                               double max_dV) 

__device__ double set_time_step (double TIME, double time_point, double max_time_step, double *CONSTANTS, double *RATES, int offset) {
 double min_time_step = 0.005;
 double min_dV = 0.2;
 double max_dV = 0.8;
 double time_step = min_time_step;
 
 if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[(208 * offset) + BCL]) * CONSTANTS[(208 * offset) + BCL]) <= time_point) {
    //printf("TIME <= time_point ms\n");
    return time_step;
    //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
  }
  else {
    //printf("TIME > time_point ms\n");
    if (std::abs(RATES[(49 * offset) + V] * time_step) <= min_dV) {//Slow changes in V
        //printf("dV/dt <= 0.2\n");
        time_step = std::abs(max_dV / RATES[(49 * offset) + V]);
        //Make sure time_step is between min time step and max_time_step
        if (time_step < min_time_step) {
            time_step = min_time_step;
        }
        else if (time_step > max_time_step) {
            time_step = max_time_step;
        }
        //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
    }
    else if (std::abs(RATES[(49 * offset) + V] * time_step) >= max_dV) {//Fast changes in V
        //printf("dV/dt >= 0.8\n");
        time_step = std::abs(min_dV / RATES[(49 * offset) + V]);
        //Make sure time_step is not less than 0.005
        if (time_step < min_time_step) {
            time_step = min_time_step;
        }
        //printf("TIME = %E, dV = %E, time_step = %E\n",TIME, RATES[V] * time_step, time_step);
    } else {
        time_step = min_time_step;
    }
    return time_step;
  }
}