/* Generated by Yosys 0.12 (git sha1 UNKNOWN, gcc 12.1.1 -march=x86-64 -mtune=generic -O2 -fno-plt -fexceptions -fstack-clash-protection -fcf-protection -fPIC -Os) */

module relu(g_input, e_input, o);
  wire _000_;
  wire _001_;
  wire _002_;
  wire _003_;
  wire _004_;
  wire _005_;
  wire _006_;
  wire _007_;
  wire _008_;
  wire _009_;
  wire _010_;
  wire _011_;
  wire _012_;
  wire _013_;
  wire _014_;
  wire _015_;
  wire _016_;
  wire _017_;
  wire _018_;
  wire _019_;
  wire _020_;
  wire _021_;
  wire _022_;
  wire _023_;
  wire _024_;
  wire _025_;
  wire _026_;
  wire _027_;
  wire _028_;
  wire _029_;
  wire _030_;
  wire _031_;
  wire _032_;
  wire _033_;
  wire _034_;
  wire _035_;
  wire _036_;
  wire _037_;
  wire _038_;
  wire _039_;
  wire _040_;
  wire _041_;
  wire _042_;
  wire _043_;
  wire _044_;
  wire _045_;
  wire _046_;
  wire _047_;
  wire _048_;
  wire _049_;
  wire _050_;
  wire _051_;
  wire _052_;
  wire _053_;
  wire _054_;
  wire _055_;
  wire _056_;
  wire _057_;
  wire _058_;
  wire _059_;
  wire _060_;
  wire _061_;
  wire _062_;
  wire _063_;
  wire _064_;
  wire _065_;
  wire _066_;
  wire _067_;
  wire _068_;
  wire _069_;
  wire _070_;
  wire _071_;
  wire _072_;
  wire _073_;
  wire _074_;
  wire _075_;
  wire _076_;
  wire _077_;
  wire _078_;
  wire _079_;
  wire _080_;
  wire _081_;
  wire _082_;
  wire _083_;
  wire _084_;
  wire _085_;
  wire _086_;
  wire _087_;
  wire _088_;
  wire _089_;
  wire _090_;
  wire _091_;
  wire _092_;
  wire _093_;
  wire _094_;
  wire _095_;
  wire _096_;
  wire _097_;
  wire _098_;
  wire _099_;
  wire _100_;
  wire _101_;
  wire _102_;
  wire _103_;
  wire _104_;
  wire _105_;
  wire _106_;
  wire _107_;
  wire _108_;
  wire _109_;
  wire _110_;
  wire _111_;
  wire _112_;
  wire _113_;
  wire _114_;
  wire _115_;
  wire _116_;
  wire _117_;
  wire _118_;
  wire _119_;
  wire _120_;
  wire _121_;
  wire _122_;
  wire _123_;
  wire _124_;
  wire _125_;
  wire _126_;
  input [63:0] e_input;
  input [63:0] g_input;
  output [63:0] o;
  ANDN _127_ (
    .A(_015_),
    .B(_059_),
    .Z(_079_)
  );
  ANDN _128_ (
    .A(_016_),
    .B(_059_),
    .Z(_080_)
  );
  ANDN _129_ (
    .A(_017_),
    .B(_059_),
    .Z(_081_)
  );
  ANDN _130_ (
    .A(_018_),
    .B(_059_),
    .Z(_082_)
  );
  ANDN _131_ (
    .A(_019_),
    .B(_059_),
    .Z(_083_)
  );
  ANDN _132_ (
    .A(_020_),
    .B(_059_),
    .Z(_084_)
  );
  ANDN _133_ (
    .A(_021_),
    .B(_059_),
    .Z(_085_)
  );
  ANDN _134_ (
    .A(_023_),
    .B(_059_),
    .Z(_087_)
  );
  ANDN _135_ (
    .A(_024_),
    .B(_059_),
    .Z(_088_)
  );
  ANDN _136_ (
    .A(_025_),
    .B(_059_),
    .Z(_089_)
  );
  ANDN _137_ (
    .A(_026_),
    .B(_059_),
    .Z(_090_)
  );
  ANDN _138_ (
    .A(_027_),
    .B(_059_),
    .Z(_091_)
  );
  ANDN _139_ (
    .A(_028_),
    .B(_059_),
    .Z(_092_)
  );
  ANDN _140_ (
    .A(_029_),
    .B(_059_),
    .Z(_093_)
  );
  ANDN _141_ (
    .A(_030_),
    .B(_059_),
    .Z(_094_)
  );
  ANDN _142_ (
    .A(_031_),
    .B(_059_),
    .Z(_095_)
  );
  ANDN _143_ (
    .A(_032_),
    .B(_059_),
    .Z(_096_)
  );
  ANDN _144_ (
    .A(_034_),
    .B(_059_),
    .Z(_098_)
  );
  ANDN _145_ (
    .A(_035_),
    .B(_059_),
    .Z(_099_)
  );
  ANDN _146_ (
    .A(_036_),
    .B(_059_),
    .Z(_100_)
  );
  ANDN _147_ (
    .A(_037_),
    .B(_059_),
    .Z(_101_)
  );
  ANDN _148_ (
    .A(_038_),
    .B(_059_),
    .Z(_102_)
  );
  ANDN _149_ (
    .A(_039_),
    .B(_059_),
    .Z(_103_)
  );
  ANDN _150_ (
    .A(_040_),
    .B(_059_),
    .Z(_104_)
  );
  ANDN _151_ (
    .A(_041_),
    .B(_059_),
    .Z(_105_)
  );
  ANDN _152_ (
    .A(_042_),
    .B(_059_),
    .Z(_106_)
  );
  ANDN _153_ (
    .A(_043_),
    .B(_059_),
    .Z(_107_)
  );
  ANDN _154_ (
    .A(_045_),
    .B(_059_),
    .Z(_109_)
  );
  ANDN _155_ (
    .A(_046_),
    .B(_059_),
    .Z(_110_)
  );
  ANDN _156_ (
    .A(_047_),
    .B(_059_),
    .Z(_111_)
  );
  ANDN _157_ (
    .A(_048_),
    .B(_059_),
    .Z(_112_)
  );
  ANDN _158_ (
    .A(_049_),
    .B(_059_),
    .Z(_113_)
  );
  ANDN _159_ (
    .A(_050_),
    .B(_059_),
    .Z(_114_)
  );
  ANDN _160_ (
    .A(_051_),
    .B(_059_),
    .Z(_115_)
  );
  ANDN _161_ (
    .A(_052_),
    .B(_059_),
    .Z(_116_)
  );
  ANDN _162_ (
    .A(_053_),
    .B(_059_),
    .Z(_117_)
  );
  ANDN _163_ (
    .A(_054_),
    .B(_059_),
    .Z(_118_)
  );
  ANDN _164_ (
    .A(_056_),
    .B(_059_),
    .Z(_120_)
  );
  ANDN _165_ (
    .A(_057_),
    .B(_059_),
    .Z(_121_)
  );
  ANDN _166_ (
    .A(_058_),
    .B(_059_),
    .Z(_122_)
  );
  ANDN _167_ (
    .A(_000_),
    .B(_059_),
    .Z(_064_)
  );
  ANDN _168_ (
    .A(_011_),
    .B(_059_),
    .Z(_075_)
  );
  ANDN _169_ (
    .A(_022_),
    .B(_059_),
    .Z(_086_)
  );
  ANDN _170_ (
    .A(_033_),
    .B(_059_),
    .Z(_097_)
  );
  ANDN _171_ (
    .A(_044_),
    .B(_059_),
    .Z(_108_)
  );
  ANDN _172_ (
    .A(_055_),
    .B(_059_),
    .Z(_119_)
  );
  ANDN _173_ (
    .A(_060_),
    .B(_059_),
    .Z(_123_)
  );
  ANDN _174_ (
    .A(_061_),
    .B(_059_),
    .Z(_124_)
  );
  ANDN _175_ (
    .A(_062_),
    .B(_059_),
    .Z(_125_)
  );
  ANDN _176_ (
    .A(_063_),
    .B(_059_),
    .Z(_126_)
  );
  ANDN _177_ (
    .A(_001_),
    .B(_059_),
    .Z(_065_)
  );
  ANDN _178_ (
    .A(_002_),
    .B(_059_),
    .Z(_066_)
  );
  ANDN _179_ (
    .A(_003_),
    .B(_059_),
    .Z(_067_)
  );
  ANDN _180_ (
    .A(_004_),
    .B(_059_),
    .Z(_068_)
  );
  ANDN _181_ (
    .A(_005_),
    .B(_059_),
    .Z(_069_)
  );
  ANDN _182_ (
    .A(_006_),
    .B(_059_),
    .Z(_070_)
  );
  ANDN _183_ (
    .A(_007_),
    .B(_059_),
    .Z(_071_)
  );
  ANDN _184_ (
    .A(_008_),
    .B(_059_),
    .Z(_072_)
  );
  ANDN _185_ (
    .A(_009_),
    .B(_059_),
    .Z(_073_)
  );
  ANDN _186_ (
    .A(_010_),
    .B(_059_),
    .Z(_074_)
  );
  ANDN _187_ (
    .A(_012_),
    .B(_059_),
    .Z(_076_)
  );
  ANDN _188_ (
    .A(_013_),
    .B(_059_),
    .Z(_077_)
  );
  ANDN _189_ (
    .A(_014_),
    .B(_059_),
    .Z(_078_)
  );
  assign o[63] = 1'b0;
  assign _015_ = e_input[23];
  assign _059_ = e_input[63];
  assign o[23] = _079_;
  assign _016_ = e_input[24];
  assign o[24] = _080_;
  assign _017_ = e_input[25];
  assign o[25] = _081_;
  assign _018_ = e_input[26];
  assign o[26] = _082_;
  assign _019_ = e_input[27];
  assign o[27] = _083_;
  assign _020_ = e_input[28];
  assign o[28] = _084_;
  assign _021_ = e_input[29];
  assign o[29] = _085_;
  assign _023_ = e_input[30];
  assign o[30] = _087_;
  assign _024_ = e_input[31];
  assign o[31] = _088_;
  assign _025_ = e_input[32];
  assign o[32] = _089_;
  assign _026_ = e_input[33];
  assign o[33] = _090_;
  assign _027_ = e_input[34];
  assign o[34] = _091_;
  assign _028_ = e_input[35];
  assign o[35] = _092_;
  assign _029_ = e_input[36];
  assign o[36] = _093_;
  assign _030_ = e_input[37];
  assign o[37] = _094_;
  assign _031_ = e_input[38];
  assign o[38] = _095_;
  assign _032_ = e_input[39];
  assign o[39] = _096_;
  assign _034_ = e_input[40];
  assign o[40] = _098_;
  assign _035_ = e_input[41];
  assign o[41] = _099_;
  assign _036_ = e_input[42];
  assign o[42] = _100_;
  assign _037_ = e_input[43];
  assign o[43] = _101_;
  assign _038_ = e_input[44];
  assign o[44] = _102_;
  assign _039_ = e_input[45];
  assign o[45] = _103_;
  assign _040_ = e_input[46];
  assign o[46] = _104_;
  assign _041_ = e_input[47];
  assign o[47] = _105_;
  assign _042_ = e_input[48];
  assign o[48] = _106_;
  assign _043_ = e_input[49];
  assign o[49] = _107_;
  assign _045_ = e_input[50];
  assign o[50] = _109_;
  assign _046_ = e_input[51];
  assign o[51] = _110_;
  assign _047_ = e_input[52];
  assign o[52] = _111_;
  assign _048_ = e_input[53];
  assign o[53] = _112_;
  assign _049_ = e_input[54];
  assign o[54] = _113_;
  assign _050_ = e_input[55];
  assign o[55] = _114_;
  assign _051_ = e_input[56];
  assign o[56] = _115_;
  assign _052_ = e_input[57];
  assign o[57] = _116_;
  assign _053_ = e_input[58];
  assign o[58] = _117_;
  assign _054_ = e_input[59];
  assign o[59] = _118_;
  assign _056_ = e_input[60];
  assign o[60] = _120_;
  assign _057_ = e_input[61];
  assign o[61] = _121_;
  assign _058_ = e_input[62];
  assign o[62] = _122_;
  assign _000_ = e_input[0];
  assign o[0] = _064_;
  assign _011_ = e_input[1];
  assign o[1] = _075_;
  assign _022_ = e_input[2];
  assign o[2] = _086_;
  assign _033_ = e_input[3];
  assign o[3] = _097_;
  assign _044_ = e_input[4];
  assign o[4] = _108_;
  assign _055_ = e_input[5];
  assign o[5] = _119_;
  assign _060_ = e_input[6];
  assign o[6] = _123_;
  assign _061_ = e_input[7];
  assign o[7] = _124_;
  assign _062_ = e_input[8];
  assign o[8] = _125_;
  assign _063_ = e_input[9];
  assign o[9] = _126_;
  assign _001_ = e_input[10];
  assign o[10] = _065_;
  assign _002_ = e_input[11];
  assign o[11] = _066_;
  assign _003_ = e_input[12];
  assign o[12] = _067_;
  assign _004_ = e_input[13];
  assign o[13] = _068_;
  assign _005_ = e_input[14];
  assign o[14] = _069_;
  assign _006_ = e_input[15];
  assign o[15] = _070_;
  assign _007_ = e_input[16];
  assign o[16] = _071_;
  assign _008_ = e_input[17];
  assign o[17] = _072_;
  assign _009_ = e_input[18];
  assign o[18] = _073_;
  assign _010_ = e_input[19];
  assign o[19] = _074_;
  assign _012_ = e_input[20];
  assign o[20] = _076_;
  assign _013_ = e_input[21];
  assign o[21] = _077_;
  assign _014_ = e_input[22];
  assign o[22] = _078_;
endmodule
