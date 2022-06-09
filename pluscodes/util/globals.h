#ifndef globals
#define globals
#include "math.h"

const unsigned char GRID_COLS = 4;

const char SEP = '+';
const unsigned char SEP_POS = 8;
const char *ALPHABET = "23456789CFGHJMPQRVWX";
const char PAD = '0';
const unsigned char BASE = 20;
const unsigned char MAX_LAT = 90;
const unsigned char MAX_LON = 180;

const unsigned char MAX_DIGITS = 15;

const unsigned char PAIR_LENGTH = 10;

const unsigned char GRID_LENGTH = MAX_DIGITS - PAIR_LENGTH;

const unsigned char GRID_COLUMNS = 4;
const unsigned char GRID_ROWS = 5;

const unsigned char MIN_TRIMMABLE_CODE_LEN = 6;
const double GRID_SIZE_DEGREES;
const int DEBUG = 5;
const double GRID_SIZE_DEGREES = 0.000125;

#define PAIR_FIRST_VALUE pow(BASE, PAIR_LENGTH / 2 - 1)
#define PAIR_PRECISION pow(BASE, 3)
#define GRID_LAT_FIRST_PLACE_VALUE pow(GRID_ROWS, GRID_LENGTH - 1)
#define GRID_LON_FIRST_PLACE_VALUE pow(GRID_COLUMNS, GRID_LENGTH - 1)
#define FINAL_LAT_PRECISION pow(PAIR_PRECISION * GRID_ROWS, MAX_DIGITS - PAIR_LENGTH)
#define FINAL_LON_PRECISION pow(PAIR_PRECISION * GRID_COLUMNS, MAX_DIGITS - PAIR_LENGTH)

#endif