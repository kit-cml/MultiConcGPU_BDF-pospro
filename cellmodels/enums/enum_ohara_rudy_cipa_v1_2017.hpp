#ifndef EN_OHARA_RUDY_CIPA_V1_2017_HPP
#define EN_OHARA_RUDY_CIPA_V1_2017_HPP

enum E_ALGEBRAIC_T{
  Istim = 0,
  mss = 1,
  hss = 2,
  hLss = 3,
  hLssp = 4,
  ass = 5,
  iss = 6,
  dss = 7,
  fss = 8,
  km2n = 9,
  xs1ss = 10,
  xk1ss = 11,
  vfrt = 12,
  tm = 13,
  thf = 14,
  ths = 15,
  jss = 16,
  ta = 17,
  delta_epi = 18,
  fcass = 19,
  anca = 20,
  td = 21,
  tff = 22,
  tfs = 23,
  xs2ss = 24,
  txs1 = 25,
  txk1 = 26,
  tj = 27,
  hssp = 28,
  mLss = 29,
  tiF_b = 30,
  assp = 31,
  tfcaf = 32,
  tfcas = 33,
  tffp = 34,
  txs2 = 35,
  CaMKb = 36,
  thsp = 37,
  tjp = 38,
  tmL = 39,
  tiS_b = 40,
  tfcafp = 41,
  CaMKa = 42,
  tiF = 43,
  Bcai = 44,
  tiS = 45,
  Bcass = 46,
  dti_develop = 47,
  Bcajsr = 48,
  dti_recover = 49,
  ENa = 50,
  tiFp = 51,
  tiSp = 52,
  EK = 53,
  EKs = 54,
  h = 55,
  hp = 56,
  fINap = 57,
  INa = 58,
  fINaLp = 59,
  INaL = 60,
  AiF = 61,
  AiS = 62,
  i = 63,
  ip = 64,
  fItop = 65,
  Ito = 66,
  f = 67,
  Afcaf = 68,
  Afcas = 69,
  fca = 70,
  fp = 71,
  fcap = 72,
  A_1 = 73,
  U_1 = 74,
  PhiCaL = 75,
  A_2 = 76,
  U_2 = 77,
  PhiCaNa = 78,
  A_3 = 79,
  U_3 = 80,
  PhiCaK = 81,
  fICaLp = 82,
  ICaL = 83,
  ICaNa = 84,
  Jrel_inf_temp = 85,
  Jrel_temp = 86,
  ICaK = 87,
  Jrel_inf = 88,
  Jrel_infp = 89,
  IKr = 90,
  tau_rel_temp = 91,
  tau_relp_temp = 92,
  KsCa = 93,
  tau_rel = 94,
  tau_relp = 95,
  IKs = 96,
  rk1 = 97,
  IK1 = 98,
  hca = 99,
  hna = 100,
  h1_i = 101,
  h2_i = 102,
  h3_i = 103,
  h4_i = 104,
  h5_i = 105,
  h6_i = 106,
  h7_i = 107,
  h8_i = 108,
  h9_i = 109,
  k3p_i = 110,
  k3pp_i = 111,
  k3_i = 112,
  k4p_i = 113,
  k4pp_i = 114,
  k4_i = 115,
  k6_i = 116,
  k7_i = 117,
  k8_i = 118,
  x1_i = 119,
  x2_i = 120,
  x3_i = 121,
  x4_i = 122,
  E1_i = 123,
  E2_i = 124,
  E3_i = 125,
  E4_i = 126,
  allo_i = 127,
  JncxNa_i = 128,
  JncxCa_i = 129,
  INaCa_i = 130,
  h1_ss = 131,
  h2_ss = 132,
  h3_ss = 133,
  h4_ss = 134,
  h5_ss = 135,
  h6_ss = 136,
  h7_ss = 137,
  h8_ss = 138,
  h9_ss = 139,
  k3p_ss = 140,
  k3pp_ss = 141,
  k3_ss = 142,
  k4p_ss = 143,
  k4pp_ss = 144,
  k4_ss = 145,
  k6_ss = 146,
  k7_ss = 147,
  k8_ss = 148,
  x1_ss = 149,
  x2_ss = 150,
  x3_ss = 151,
  x4_ss = 152,
  E1_ss = 153,
  E2_ss = 154,
  E3_ss = 155,
  E4_ss = 156,
  allo_ss = 157,
  JncxNa_ss = 158,
  JncxCa_ss = 159,
  INaCa_ss = 160,
  Knai = 161,
  Knao = 162,
  P = 163,
  a1 = 164,
  b2 = 165,
  a3 = 166,
  b3 = 167,
  b4 = 168,
  x1 = 169,
  x2 = 170,
  x3 = 171,
  x4 = 172,
  E1 = 173,
  E2 = 174,
  E3 = 175,
  E4 = 176,
  JnakNa = 177,
  JnakK = 178,
  INaK = 179,
  xkb = 180,
  IKb = 181,
  A_Nab = 182,
  JdiffK = 183,
  U_Nab = 184,
  INab = 185,
  A_Cab = 186,
  JdiffNa = 187,
  U_Cab = 188,
  ICab = 189,
  IpCa = 190,
  Jdiff = 191,
  fJrelp = 192,
  Jrel = 193,
  Jupnp = 194,
  Jupp = 195,
  fJupp = 196,
  Jleak = 197,
  Jup = 198,
  Jtr = 199,
};


enum E_CONSTANTS_T{
  celltype = 0,
  nao = 1,
  cao = 2,
  ko = 3,
  R = 4,
  T = 5,
  F = 6,
  zna = 7,
  zca = 8,
  zk = 9,
  L = 10,
  rad = 11,
  stim_start = 12,
  stim_end = 13,
  amp = 14,
  BCL = 15,
  duration = 16,
  KmCaMK = 17,
  aCaMK = 18,
  bCaMK = 19,
  CaMKo = 20,
  KmCaM = 21,
  cmdnmax_b = 22,
  kmcmdn = 23,
  trpnmax = 24,
  kmtrpn = 25,
  BSRmax = 26,
  KmBSR = 27,
  BSLmax = 28,
  KmBSL = 29,
  csqnmax = 30,
  kmcsqn = 31,
  cm = 32,
  PKNa = 33,
  mssV1 = 34,
  mssV2 = 35,
  mtV1 = 36,
  mtV2 = 37,
  mtD1 = 38,
  mtD2 = 39,
  mtV3 = 40,
  mtV4 = 41,
  hssV1 = 42,
  hssV2 = 43,
  Ahf = 44,
  GNa = 45,
  shift_INa_inact = 46,
  thL = 47,
  GNaL_b = 48,
  Gto_b = 49,
  Kmn = 50,
  k2n = 51,
  PCa_b = 52,
  GKr_b = 53,
  A1 = 54,
  B1 = 55,
  q1 = 56,
  A2 = 57,
  B2 = 58,
  q2 = 59,
  A3 = 60,
  B3 = 61,
  q3 = 62,
  A4 = 63,
  B4 = 64,
  q4 = 65,
  A11 = 66,
  B11 = 67,
  q11 = 68,
  A21 = 69,
  B21 = 70,
  q21 = 71,
  A31 = 72,
  B31 = 73,
  q31 = 74,
  A41 = 75,
  B41 = 76,
  q41 = 77,
  A51 = 78,
  B51 = 79,
  q51 = 80,
  A52 = 81,
  B52 = 82,
  q52 = 83,
  A53 = 84,
  B53 = 85,
  q53 = 86,
  A61 = 87,
  B61 = 88,
  q61 = 89,
  A62 = 90,
  B62 = 91,
  q62 = 92,
  A63 = 93,
  B63 = 94,
  q63 = 95,
  Kmax = 96,
  Ku = 97,
  n = 98,
  halfmax = 99,
  Kt = 100,
  Vhalf = 101,
  Temp = 102,
  GKs_b = 103,
  txs1_max = 104,
  GK1_b = 105,
  kna1 = 106,
  kna2 = 107,
  kna3 = 108,
  kasymm = 109,
  wna = 110,
  wca = 111,
  wnaca = 112,
  kcaon = 113,
  kcaoff = 114,
  qna = 115,
  qca = 116,
  KmCaAct = 117,
  Gncx_b = 118,
  k1p = 119,
  k1m = 120,
  k2p = 121,
  k2m = 122,
  k3p = 123,
  k3m = 124,
  k4p = 125,
  k4m = 126,
  Knai0 = 127,
  Knao0 = 128,
  delta = 129,
  Kki = 130,
  Kko = 131,
  MgADP = 132,
  MgATP = 133,
  Kmgatp = 134,
  H = 135,
  eP = 136,
  Khp = 137,
  Knap = 138,
  Kxkur = 139,
  Pnak_b = 140,
  GKb_b = 141,
  PNab = 142,
  PCab = 143,
  GpCa = 144,
  KmCap = 145,
  bt = 146,
  Jrel_scaling_factor = 147,
  Jup_b = 148,
  frt = 149,
  cmdnmax = 150,
  Ahs = 151,
  thLp = 152,
  GNaL = 153,
  Gto = 154,
  Aff = 155,
  PCa = 156,
  tjca = 157,
  v0_CaL = 158,
  GKr = 159,
  GKs = 160,
  GK1 = 161,
  vcell = 162,
  GKb = 163,
  v0_Nab = 164,
  v0_Cab = 165,
  a_rel = 166,
  btp = 167,
  upScale = 168,
  ffrt = 169,
  Afs = 170,
  PCap = 171,
  PCaNa = 172,
  PCaK = 173,
  B_1 = 174,
  B_2 = 175,
  B_3 = 176,
  Ageo = 177,
  B_Nab = 178,
  B_Cab = 179,
  a_relp = 180,
  PCaNap = 181,
  PCaKp = 182,
  Acap = 183,
  vmyo = 184,
  vnsr = 185,
  vjsr = 186,
  vss = 187,
  h10_i = 188,
  h11_i = 189,
  h12_i = 190,
  k1_i = 191,
  k2_i = 192,
  k5_i = 193,
  Gncx = 194,
  h10_ss = 195,
  h11_ss = 196,
  h12_ss = 197,
  k1_ss = 198,
  k2_ss = 199,
  k5_ss = 200,
  b1 = 201,
  a2 = 202,
  a4 = 203,
  Pnak = 204,
  cnc = 205,
};


enum E_STATES_T{
  V = 0,
  CaMKt = 1,
  cass = 2,
  nai = 3,
  nass = 4,
  ki = 5,
  kss = 6,
  cansr = 7,
  cajsr = 8,
  cai = 9,
  m = 10,
  hf = 11,
  hs = 12,
  j = 13,
  hsp = 14,
  jp = 15,
  mL = 16,
  hL = 17,
  hLp = 18,
  a = 19,
  iF = 20,
  iS = 21,
  ap = 22,
  iFp = 23,
  iSp = 24,
  d = 25,
  ff = 26,
  fs = 27,
  fcaf = 28,
  fcas = 29,
  jca = 30,
  ffp = 31,
  fcafp = 32,
  nca = 33,
  IC1 = 34,
  IC2 = 35,
  C1 = 36,
  C2 = 37,
  O = 38,
  IO = 39,
  IObound = 40,
  Obound = 41,
  Cbound = 42,
  D = 43,
  xs1 = 44,
  xs2 = 45,
  xk1 = 46,
  Jrelnp = 47,
  Jrelp = 48,
};


#endif