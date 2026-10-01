// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#include "common.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <random>
#include <algorithm>
#include <string>
#include <cstring>
#include <fstream>
#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <limits>
#include <queue>
#include <regex>
#include <cstdio>

#ifdef DEBUG
    struct debug_notifier{
        debug_notifier(){
            std::printf("WARNING: DEBUG MODE\n");
        }
    };
    static debug_notifier Debug_notifier;
#endif