//#ifndef globals
//#define globals

#include "math.h"
#include "stddef.h"

static const  unsigned char GRID_COLS = 4;

static const  char SEP = '+';
static const  char *ALPHABET = "23456789CFGHJMPQRVWX";
static const  char PAD = '0';

static const  size_t SEP_POS = 8;
static const  size_t BASE = 20;
static const  size_t MAX_DIGITS = 15;
static const  size_t PAIR_LENGTH = 10;
static const  size_t GRID_LENGTH = MAX_DIGITS - PAIR_LENGTH;
static const  size_t L = GRID_LENGTH;
//#define L (MAX_DIGITS - PAIR_LENGTH + 1)
static const  size_t GRID_COLUMNS = 4;
static const  size_t GRID_ROWS = 5;
static const  size_t MIN_TRIMMABLE_CODE_LEN = 6;

static const  int MAX_LAT = 90;
static const  int MAX_LON = 180;
static const  double GRID_SIZE_DEGREES = 0.000125;

static const int DEBUG = 5;

//extern unsigned long PAIR_PRECISION_FIRST_VALUE;
//extern unsigned long PAIR_PRECISION;
//extern unsigned long GRID_LAT_FIRST_PLACE_VALUE;
//extern unsigned long GRID_LON_FIRST_PLACE_VALUE;
//extern unsigned long FINAL_LAT_PRECISION;
//extern unsigned long FINAL_LON_PRECISION;
//extern double GRID_LAT_RESOLUTION;
//extern double GRID_LON_RESOLUTION;

 #define PAIR_PRECISION_FIRST_VALUE pow(BASE, PAIR_LENGTH / 2 - 1)
 #define PAIR_PRECISION pow(BASE, 3)
 #define GRID_LAT_FIRST_PLACE_VALUE pow(GRID_ROWS, GRID_LENGTH - 1)
 #define GRID_LON_FIRST_PLACE_VALUE pow(GRID_COLUMNS, GRID_LENGTH - 1)
 #define FINAL_LAT_PRECISION pow(GRID_ROWS, GRID_LENGTH) * PAIR_PRECISION
 #define FINAL_LON_PRECISION pow(GRID_COLUMNS, GRID_LENGTH) * PAIR_PRECISION
 #define GRID_LAT_RESOLUTION pow(GRID_ROWS, GRID_LENGTH)
 #define GRID_LON_RESOLUTION pow(GRID_COLUMNS, GRID_LENGTH)
