//===- TableGen'erated file -------------------------------------*- C++ -*-===//
//
// Target Instruction Enum Values
//
// Automatically generated file, do not edit!
//
//===----------------------------------------------------------------------===//

namespace llvm {

namespace PTX {
  enum {
    PHI	= 0,
    INLINEASM	= 1,
    PROLOG_LABEL	= 2,
    EH_LABEL	= 3,
    GC_LABEL	= 4,
    KILL	= 5,
    EXTRACT_SUBREG	= 6,
    INSERT_SUBREG	= 7,
    IMPLICIT_DEF	= 8,
    SUBREG_TO_REG	= 9,
    COPY_TO_REGCLASS	= 10,
    DBG_VALUE	= 11,
    REG_SEQUENCE	= 12,
    COPY	= 13,
    ADDri16	= 14,
    ADDri32	= 15,
    ADDri64	= 16,
    ADDrr16	= 17,
    ADDrr32	= 18,
    ADDrr64	= 19,
    EXIT	= 20,
    FADDri32	= 21,
    FADDri64	= 22,
    FADDrr32	= 23,
    FADDrr64	= 24,
    FMULri32	= 25,
    FMULri64	= 26,
    FMULrr32	= 27,
    FMULrr64	= 28,
    FSUBri32	= 29,
    FSUBri64	= 30,
    FSUBrr32	= 31,
    FSUBrr64	= 32,
    LDcf32ii32	= 33,
    LDcf32ii64	= 34,
    LDcf32ri32	= 35,
    LDcf32ri64	= 36,
    LDcf32rr32	= 37,
    LDcf32rr64	= 38,
    LDcf64ii32	= 39,
    LDcf64ii64	= 40,
    LDcf64ri32	= 41,
    LDcf64ri64	= 42,
    LDcf64rr32	= 43,
    LDcf64rr64	= 44,
    LDcu16ii32	= 45,
    LDcu16ii64	= 46,
    LDcu16ri32	= 47,
    LDcu16ri64	= 48,
    LDcu16rr32	= 49,
    LDcu16rr64	= 50,
    LDcu32ii32	= 51,
    LDcu32ii64	= 52,
    LDcu32ri32	= 53,
    LDcu32ri64	= 54,
    LDcu32rr32	= 55,
    LDcu32rr64	= 56,
    LDcu64ii32	= 57,
    LDcu64ii64	= 58,
    LDcu64ri32	= 59,
    LDcu64ri64	= 60,
    LDcu64rr32	= 61,
    LDcu64rr64	= 62,
    LDgf32ii32	= 63,
    LDgf32ii64	= 64,
    LDgf32ri32	= 65,
    LDgf32ri64	= 66,
    LDgf32rr32	= 67,
    LDgf32rr64	= 68,
    LDgf64ii32	= 69,
    LDgf64ii64	= 70,
    LDgf64ri32	= 71,
    LDgf64ri64	= 72,
    LDgf64rr32	= 73,
    LDgf64rr64	= 74,
    LDgu16ii32	= 75,
    LDgu16ii64	= 76,
    LDgu16ri32	= 77,
    LDgu16ri64	= 78,
    LDgu16rr32	= 79,
    LDgu16rr64	= 80,
    LDgu32ii32	= 81,
    LDgu32ii64	= 82,
    LDgu32ri32	= 83,
    LDgu32ri64	= 84,
    LDgu32rr32	= 85,
    LDgu32rr64	= 86,
    LDgu64ii32	= 87,
    LDgu64ii64	= 88,
    LDgu64ri32	= 89,
    LDgu64ri64	= 90,
    LDgu64rr32	= 91,
    LDgu64rr64	= 92,
    LDlf32ii32	= 93,
    LDlf32ii64	= 94,
    LDlf32ri32	= 95,
    LDlf32ri64	= 96,
    LDlf32rr32	= 97,
    LDlf32rr64	= 98,
    LDlf64ii32	= 99,
    LDlf64ii64	= 100,
    LDlf64ri32	= 101,
    LDlf64ri64	= 102,
    LDlf64rr32	= 103,
    LDlf64rr64	= 104,
    LDlu16ii32	= 105,
    LDlu16ii64	= 106,
    LDlu16ri32	= 107,
    LDlu16ri64	= 108,
    LDlu16rr32	= 109,
    LDlu16rr64	= 110,
    LDlu32ii32	= 111,
    LDlu32ii64	= 112,
    LDlu32ri32	= 113,
    LDlu32ri64	= 114,
    LDlu32rr32	= 115,
    LDlu32rr64	= 116,
    LDlu64ii32	= 117,
    LDlu64ii64	= 118,
    LDlu64ri32	= 119,
    LDlu64ri64	= 120,
    LDlu64rr32	= 121,
    LDlu64rr64	= 122,
    LDpiF32	= 123,
    LDpiF64	= 124,
    LDpiU16	= 125,
    LDpiU32	= 126,
    LDpiU64	= 127,
    LDsf32ii32	= 128,
    LDsf32ii64	= 129,
    LDsf32ri32	= 130,
    LDsf32ri64	= 131,
    LDsf32rr32	= 132,
    LDsf32rr64	= 133,
    LDsf64ii32	= 134,
    LDsf64ii64	= 135,
    LDsf64ri32	= 136,
    LDsf64ri64	= 137,
    LDsf64rr32	= 138,
    LDsf64rr64	= 139,
    LDsu16ii32	= 140,
    LDsu16ii64	= 141,
    LDsu16ri32	= 142,
    LDsu16ri64	= 143,
    LDsu16rr32	= 144,
    LDsu16rr64	= 145,
    LDsu32ii32	= 146,
    LDsu32ii64	= 147,
    LDsu32ri32	= 148,
    LDsu32ri64	= 149,
    LDsu32rr32	= 150,
    LDsu32rr64	= 151,
    LDsu64ii32	= 152,
    LDsu64ii64	= 153,
    LDsu64ri32	= 154,
    LDsu64ri64	= 155,
    LDsu64rr32	= 156,
    LDsu64rr64	= 157,
    MOVF32ri	= 158,
    MOVF32rr	= 159,
    MOVF64ri	= 160,
    MOVF64rr	= 161,
    MOVPREDri	= 162,
    MOVPREDrr	= 163,
    MOVU164ri	= 164,
    MOVU16ri	= 165,
    MOVU16rr	= 166,
    MOVU32ri	= 167,
    MOVU32rr	= 168,
    MOVU64rr	= 169,
    PTX_BAR_SYNC	= 170,
    PTX_READ_TID_R64	= 171,
    PTX_READ_TID_W	= 172,
    PTX_READ_TID_X	= 173,
    PTX_READ_TID_Y	= 174,
    PTX_READ_TID_Z	= 175,
    RET	= 176,
    SHLir	= 177,
    SHLri	= 178,
    SHLrr	= 179,
    SRAir	= 180,
    SRAri	= 181,
    SRArr	= 182,
    SRLir	= 183,
    SRLri	= 184,
    SRLrr	= 185,
    STgf32ii32	= 186,
    STgf32ii64	= 187,
    STgf32ri32	= 188,
    STgf32ri64	= 189,
    STgf32rr32	= 190,
    STgf32rr64	= 191,
    STgf64ii32	= 192,
    STgf64ii64	= 193,
    STgf64ri32	= 194,
    STgf64ri64	= 195,
    STgf64rr32	= 196,
    STgf64rr64	= 197,
    STgu16ii32	= 198,
    STgu16ii64	= 199,
    STgu16ri32	= 200,
    STgu16ri64	= 201,
    STgu16rr32	= 202,
    STgu16rr64	= 203,
    STgu32ii32	= 204,
    STgu32ii64	= 205,
    STgu32ri32	= 206,
    STgu32ri64	= 207,
    STgu32rr32	= 208,
    STgu32rr64	= 209,
    STgu64ii32	= 210,
    STgu64ii64	= 211,
    STgu64ri32	= 212,
    STgu64ri64	= 213,
    STgu64rr32	= 214,
    STgu64rr64	= 215,
    STlf32ii32	= 216,
    STlf32ii64	= 217,
    STlf32ri32	= 218,
    STlf32ri64	= 219,
    STlf32rr32	= 220,
    STlf32rr64	= 221,
    STlf64ii32	= 222,
    STlf64ii64	= 223,
    STlf64ri32	= 224,
    STlf64ri64	= 225,
    STlf64rr32	= 226,
    STlf64rr64	= 227,
    STlu16ii32	= 228,
    STlu16ii64	= 229,
    STlu16ri32	= 230,
    STlu16ri64	= 231,
    STlu16rr32	= 232,
    STlu16rr64	= 233,
    STlu32ii32	= 234,
    STlu32ii64	= 235,
    STlu32ri32	= 236,
    STlu32ri64	= 237,
    STlu32rr32	= 238,
    STlu32rr64	= 239,
    STlu64ii32	= 240,
    STlu64ii64	= 241,
    STlu64ri32	= 242,
    STlu64ri64	= 243,
    STlu64rr32	= 244,
    STlu64rr64	= 245,
    STsf32ii32	= 246,
    STsf32ii64	= 247,
    STsf32ri32	= 248,
    STsf32ri64	= 249,
    STsf32rr32	= 250,
    STsf32rr64	= 251,
    STsf64ii32	= 252,
    STsf64ii64	= 253,
    STsf64ri32	= 254,
    STsf64ri64	= 255,
    STsf64rr32	= 256,
    STsf64rr64	= 257,
    STsu16ii32	= 258,
    STsu16ii64	= 259,
    STsu16ri32	= 260,
    STsu16ri64	= 261,
    STsu16rr32	= 262,
    STsu16rr64	= 263,
    STsu32ii32	= 264,
    STsu32ii64	= 265,
    STsu32ri32	= 266,
    STsu32ri64	= 267,
    STsu32rr32	= 268,
    STsu32rr64	= 269,
    STsu64ii32	= 270,
    STsu64ii64	= 271,
    STsu64ri32	= 272,
    STsu64ri64	= 273,
    STsu64rr32	= 274,
    STsu64rr64	= 275,
    SUBri16	= 276,
    SUBri32	= 277,
    SUBri64	= 278,
    SUBrr16	= 279,
    SUBrr32	= 280,
    SUBrr64	= 281,
    INSTRUCTION_LIST_END = 282
  };
}
} // End llvm namespace 