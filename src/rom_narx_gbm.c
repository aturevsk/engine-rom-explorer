#include "rom_narx_gbm.h"
#include <string.h>

static float gbm_tree_0(const float *x) {
    if (x[0] <= -0.0624523424f) {
        if (x[0] <= -1.2311339974f) {
            if (x[6] <= 1.0921502113f) {
                return -1.4015037950f;
            } else {
                return -2.5683077987f;
            }
        } else {
            if (x[8] <= -0.6160222888f) {
                return -0.8901372182f;
            } else {
                return -0.2793739215f;
            }
        }
    } else {
        if (x[8] <= 0.9446932673f) {
            if (x[8] <= 0.6627126038f) {
                return 0.4676994360f;
            } else {
                return 0.7912170224f;
            }
        } else {
            if (x[8] <= 1.2397320867f) {
                return 1.0786381119f;
            } else {
                return 1.3986461473f;
            }
        }
    }
}
static float gbm_tree_1(const float *x) {
    if (x[0] <= -0.0389216300f) {
        if (x[0] <= -1.2311339974f) {
            if (x[5] <= 0.7732171118f) {
                return -1.2070444599f;
            } else {
                return -2.2256637644f;
            }
        } else {
            if (x[8] <= -0.6980625093f) {
                return -0.8592151942f;
            } else {
                return -0.2856243923f;
            }
        }
    } else {
        if (x[8] <= 0.9329254627f) {
            if (x[8] <= 0.6242017150f) {
                return 0.4234173080f;
            } else {
                return 0.6926881125f;
            }
        } else {
            if (x[8] <= 1.3052196503f) {
                return 0.9875971630f;
            } else {
                return 1.3024828823f;
            }
        }
    }
}
static float gbm_tree_2(const float *x) {
    if (x[0] <= 0.0119797238f) {
        if (x[0] <= -1.3688070178f) {
            if (x[4] <= 1.3072569370f) {
                return -1.4429307743f;
            } else {
                return -2.4178946763f;
            }
        } else {
            if (x[8] <= -0.7733841240f) {
                return -0.8584319778f;
            } else {
                return -0.2752368464f;
            }
        }
    } else {
        if (x[8] <= 1.0121148825f) {
            if (x[8] <= 0.7323944271f) {
                return 0.4237219695f;
            } else {
                return 0.6910407996f;
            }
        } else {
            if (x[8] <= 1.3453978896f) {
                return 0.9372345886f;
            } else {
                return 1.2035701371f;
            }
        }
    }
}
static float gbm_tree_3(const float *x) {
    if (x[0] <= -0.1501752883f) {
        if (x[0] <= -1.3942635059f) {
            if (x[6] <= 1.6900041699f) {
                return -1.3561051805f;
            } else {
                return -2.2422016725f;
            }
        } else {
            if (x[8] <= -0.8658307493f) {
                return -0.8311374659f;
            } else {
                return -0.2918663994f;
            }
        }
    } else {
        if (x[8] <= 0.8309078217f) {
            if (x[0] <= 0.0614425540f) {
                return 0.0897601465f;
            } else {
                return 0.4230691163f;
            }
        } else {
            if (x[8] <= 1.1579714417f) {
                return 0.7103043276f;
            } else {
                return 0.9899928849f;
            }
        }
    }
}
static float gbm_tree_4(const float *x) {
    if (x[0] <= -0.1054882109f) {
        if (x[0] <= -1.2247070074f) {
            if (x[6] <= 1.2985596657f) {
                return -0.9483876294f;
            } else {
                return -1.8103716699f;
            }
        } else {
            if (x[8] <= -0.4927051812f) {
                return -0.5570086039f;
            } else {
                return -0.1344685200f;
            }
        }
    } else {
        if (x[8] <= 0.8705578744f) {
            if (x[2] <= -0.1804809943f) {
                return 0.7881582833f;
            } else {
                return 0.3245786441f;
            }
        } else {
            if (x[8] <= 1.1993450522f) {
                return 0.6694743153f;
            } else {
                return 0.9116568334f;
            }
        }
    }
}
static float gbm_tree_5(const float *x) {
    if (x[0] <= -0.0333222114f) {
        if (x[7] <= 0.7964042425f) {
            if (x[8] <= -0.4251044244f) {
                return -0.5001023169f;
            } else {
                return -0.0721783172f;
            }
        } else {
            if (x[0] <= -1.0783912539f) {
                return -1.4407782278f;
            } else {
                return -0.7382097062f;
            }
        }
    } else {
        if (x[8] <= 0.9856749475f) {
            if (x[8] <= 0.5680409372f) {
                return 0.2396890243f;
            } else {
                return 0.4634740911f;
            }
        } else {
            if (x[8] <= 1.3998084068f) {
                return 0.6861052076f;
            } else {
                return 0.9261056788f;
            }
        }
    }
}
static float gbm_tree_6(const float *x) {
    if (x[0] <= -0.0790997855f) {
        if (x[7] <= 0.9590725601f) {
            if (x[8] <= -0.9054059982f) {
                return -0.6478736579f;
            } else {
                return -0.1903124436f;
            }
        } else {
            if (x[5] <= 1.6900041699f) {
                return -1.0189569718f;
            } else {
                return -1.7747245393f;
            }
        }
    } else {
        if (x[8] <= 0.8136557341f) {
            if (x[2] <= -0.1041463614f) {
                return 0.6565054979f;
            } else {
                return 0.2427448369f;
            }
        } else {
            if (x[8] <= 1.1272981763f) {
                return 0.5039056380f;
            } else {
                return 0.7222976469f;
            }
        }
    }
}
static float gbm_tree_7(const float *x) {
    if (x[0] <= -0.2210922539f) {
        if (x[7] <= 0.5913814902f) {
            if (x[8] <= -0.3943532556f) {
                return -0.3851751071f;
            } else {
                return -0.0454123623f;
            }
        } else {
            if (x[0] <= -1.0418403745f) {
                return -1.1363002835f;
            } else {
                return -0.5250787403f;
            }
        }
    } else {
        if (x[8] <= 0.7647520900f) {
            if (x[5] <= -0.3247684538f) {
                return 0.3800188884f;
            } else {
                return 0.1345523680f;
            }
        } else {
            if (x[8] <= 1.1161348224f) {
                return 0.4421602202f;
            } else {
                return 0.6426006602f;
            }
        }
    }
}
static float gbm_tree_8(const float *x) {
    if (x[0] <= -0.2702689320f) {
        if (x[7] <= 1.2592702508f) {
            if (x[8] <= -1.0750870705f) {
                return -0.6369697123f;
            } else {
                return -0.2103199415f;
            }
        } else {
            if (x[5] <= 1.8343026042f) {
                return -0.9823166191f;
            } else {
                return -1.6150135353f;
            }
        }
    } else {
        if (x[8] <= 0.7047938108f) {
            if (x[5] <= -0.3908181190f) {
                return 0.3476054428f;
            } else {
                return 0.0891700779f;
            }
        } else {
            if (x[8] <= 1.2173155546f) {
                return 0.4113633261f;
            } else {
                return 0.6119485186f;
            }
        }
    }
}
static float gbm_tree_9(const float *x) {
    if (x[0] <= 0.0646714978f) {
        if (x[8] <= -1.2080467939f) {
            if (x[8] <= -2.4228916168f) {
                return -1.0837931878f;
            } else {
                return -0.6650189391f;
            }
        } else {
            if (x[2] <= 0.9652461112f) {
                return -0.1673481785f;
            } else {
                return -1.3719305132f;
            }
        }
    } else {
        if (x[8] <= 1.0470192432f) {
            if (x[8] <= -0.0225925725f) {
                return 0.6971554874f;
            } else {
                return 0.2561235703f;
            }
        } else {
            if (x[8] <= 1.4352864623f) {
                return 0.4661934363f;
            } else {
                return 0.6626274108f;
            }
        }
    }
}
static float gbm_tree_10(const float *x) {
    if (x[0] <= -0.6335544586f) {
        if (x[6] <= 0.3999048173f) {
            if (x[8] <= -0.9844379127f) {
                return -0.4207407061f;
            } else {
                return -0.2440669885f;
            }
        } else {
            if (x[4] <= 1.1545779705f) {
                return -0.6186282613f;
            } else {
                return -1.0430062978f;
            }
        }
    } else {
        if (x[8] <= 0.5035752654f) {
            if (x[8] <= -0.2673114687f) {
                return -0.1471231212f;
            } else {
                return 0.0737920804f;
            }
        } else {
            if (x[8] <= 1.0730030537f) {
                return 0.2866354001f;
            } else {
                return 0.4610632720f;
            }
        }
    }
}
static float gbm_tree_11(const float *x) {
    if (x[0] <= -0.3968900591f) {
        if (x[7] <= 0.4629180133f) {
            if (x[0] <= -0.7605390847f) {
                return -0.2853322606f;
            } else {
                return -0.0664667299f;
            }
        } else {
            if (x[6] <= 1.3817900419f) {
                return -0.4928220172f;
            } else {
                return -0.9179280484f;
            }
        }
    } else {
        if (x[8] <= 0.5926342309f) {
            if (x[7] <= -0.1009423286f) {
                return 0.2149860674f;
            } else {
                return -0.0127972481f;
            }
        } else {
            if (x[8] <= 1.2917686105f) {
                return 0.2979520057f;
            } else {
                return 0.4773431690f;
            }
        }
    }
}
static float gbm_tree_12(const float *x) {
    if (x[0] <= -0.6229124963f) {
        if (x[7] <= 0.6969783008f) {
            if (x[8] <= -1.0191625953f) {
                return -0.3670023291f;
            } else {
                return -0.1943930847f;
            }
        } else {
            if (x[3] <= 1.3855313063f) {
                return -0.5703611162f;
            } else {
                return -1.0542343293f;
            }
        }
    } else {
        if (x[8] <= 0.4754088372f) {
            if (x[5] <= -0.9547129571f) {
                return 0.1574907053f;
            } else {
                return -0.0542905448f;
            }
        } else {
            if (x[8] <= 1.0202061534f) {
                return 0.2195918831f;
            } else {
                return 0.3695051312f;
            }
        }
    }
}
static float gbm_tree_13(const float *x) {
    if (x[0] <= 0.0641030483f) {
        if (x[0] <= -0.9011278749f) {
            if (x[6] <= 0.7734444737f) {
                return -0.2851573260f;
            } else {
                return -0.6420771492f;
            }
        } else {
            if (x[7] <= -1.3401378989f) {
                return 0.1277877049f;
            } else {
                return -0.1156391556f;
            }
        }
    } else {
        if (x[8] <= 0.8992282152f) {
            if (x[8] <= 0.0571555411f) {
                return 0.6684375553f;
            } else {
                return 0.1422872439f;
            }
        } else {
            if (x[8] <= 1.4712534547f) {
                return 0.2912281311f;
            } else {
                return 0.4913396309f;
            }
        }
    }
}
static float gbm_tree_14(const float *x) {
    if (x[0] <= -0.5827188790f) {
        if (x[6] <= 0.2820640504f) {
            if (x[0] <= -0.9849180579f) {
                return -0.2357313526f;
            } else {
                return -0.1134280429f;
            }
        } else {
            if (x[4] <= 1.7352460027f) {
                return -0.4374032294f;
            } else {
                return -0.9950055789f;
            }
        }
    } else {
        if (x[8] <= 0.5351257026f) {
            if (x[7] <= -0.0283009931f) {
                return 0.0999325132f;
            } else {
                return -0.0614685456f;
            }
        } else {
            if (x[8] <= 1.3361089826f) {
                return 0.2157761780f;
            } else {
                return 0.3742727515f;
            }
        }
    }
}
static float gbm_tree_15(const float *x) {
    if (x[0] <= -0.3970602006f) {
        if (x[0] <= -1.4536335468f) {
            if (x[5] <= 2.0341761112f) {
                return -0.5054446744f;
            } else {
                return -0.9873965674f;
            }
        } else {
            if (x[6] <= -0.1449632645f) {
                return -0.0902638690f;
            } else {
                return -0.2615933057f;
            }
        }
    } else {
        if (x[8] <= 0.7083196938f) {
            if (x[7] <= -0.0283009931f) {
                return 0.1476699783f;
            } else {
                return 0.0094781025f;
            }
        } else {
            if (x[8] <= 1.4218315482f) {
                return 0.2173133575f;
            } else {
                return 0.3716355627f;
            }
        }
    }
}
static float gbm_tree_16(const float *x) {
    if (x[0] <= -0.6446721256f) {
        if (x[7] <= 0.3327946663f) {
            if (x[8] <= -0.8140428066f) {
                return -0.2030611403f;
            } else {
                return -0.1158323317f;
            }
        } else {
            if (x[0] <= -1.6837124825f) {
                return -0.7434967650f;
            } else {
                return -0.3555993947f;
            }
        }
    } else {
        if (x[0] <= 0.1789916977f) {
            if (x[7] <= -1.0835598111f) {
                return 0.1222987373f;
            } else {
                return -0.0692301860f;
            }
        } else {
            if (x[8] <= 0.0571555411f) {
                return 0.5270418692f;
            } else {
                return 0.1750185025f;
            }
        }
    }
}
static float gbm_tree_17(const float *x) {
    if (x[0] <= 0.0641030483f) {
        if (x[6] <= 0.7289641201f) {
            if (x[8] <= -0.3531023264f) {
                return -0.1468080231f;
            } else {
                return 0.0073387368f;
            }
        } else {
            if (x[2] <= 1.3855313063f) {
                return -0.3275950400f;
            } else {
                return -0.7592814809f;
            }
        }
    } else {
        if (x[8] <= -0.0225925725f) {
            if (x[0] <= 0.8321090937f) {
                return 0.2917564453f;
            } else {
                return 0.8310141332f;
            }
        } else {
            if (x[8] <= 0.7911322415f) {
                return 0.0672285563f;
            } else {
                return 0.2045037295f;
            }
        }
    }
}
static float gbm_tree_18(const float *x) {
    if (x[0] <= -0.6784271896f) {
        if (x[5] <= 0.8683427572f) {
            if (x[8] <= -1.3165556788f) {
                return -0.3022439202f;
            } else {
                return -0.1292822919f;
            }
        } else {
            if (x[5] <= 2.1871088743f) {
                return -0.3926111661f;
            } else {
                return -0.8619266210f;
            }
        }
    } else {
        if (x[8] <= 0.4210012704f) {
            if (x[7] <= 0.2181454673f) {
                return 0.0394672643f;
            } else {
                return -0.1027303090f;
            }
        } else {
            if (x[8] <= 1.1001962423f) {
                return 0.1177382173f;
            } else {
                return 0.2239094356f;
            }
        }
    }
}
static float gbm_tree_19(const float *x) {
    if (x[0] <= 0.0670265034f) {
        if (x[0] <= -1.0412757397f) {
            if (x[4] <= -0.1629728898f) {
                return -0.1333110284f;
            } else {
                return -0.3373294638f;
            }
        } else {
            if (x[7] <= -1.5148164034f) {
                return 0.0871763467f;
            } else {
                return -0.0777584935f;
            }
        }
    } else {
        if (x[8] <= 1.1501164436f) {
            if (x[8] <= -0.0225925725f) {
                return 0.4025051297f;
            } else {
                return 0.0935672418f;
            }
        } else {
            if (x[8] <= 1.5194897652f) {
                return 0.1834062321f;
            } else {
                return 0.3419183716f;
            }
        }
    }
}
static float gbm_tree_20(const float *x) {
    if (x[0] <= -0.7247267962f) {
        if (x[9] <= 0.2219164073f) {
            if (x[8] <= -1.4034998417f) {
                return -0.2742470690f;
            } else {
                return -0.1019092612f;
            }
        } else {
            if (x[4] <= 2.0252341032f) {
                return -0.4389206554f;
            } else {
                return -0.7791694241f;
            }
        }
    } else {
        if (x[0] <= 0.2182378992f) {
            if (x[7] <= -1.1269288659f) {
                return 0.0914097051f;
            } else {
                return -0.0521958950f;
            }
        } else {
            if (x[1] <= 0.1794634759f) {
                return 0.4056348605f;
            } else {
                return 0.1168431305f;
            }
        }
    }
}
static float gbm_tree_21(const float *x) {
    if (x[0] <= -0.7348583341f) {
        if (x[1] <= -0.3564412035f) {
            if (x[8] <= -1.4399091601f) {
                return -0.2608629045f;
            } else {
                return -0.0976999417f;
            }
        } else {
            if (x[0] <= -1.7918764353f) {
                return -0.7844412544f;
            } else {
                return -0.5215393735f;
            }
        }
    } else {
        if (x[8] <= 0.6766097844f) {
            if (x[8] <= -0.2798920870f) {
                return -0.0589338178f;
            } else {
                return 0.0351990512f;
            }
        } else {
            if (x[8] <= 1.5023001432f) {
                return 0.1215152668f;
            } else {
                return 0.2799279768f;
            }
        }
    }
}
static float gbm_tree_22(const float *x) {
    if (x[0] <= -0.5822741687f) {
        if (x[9] <= 0.1251348630f) {
            if (x[8] <= -1.2613665462f) {
                return -0.2088050264f;
            } else {
                return -0.0609320739f;
            }
        } else {
            if (x[1] <= -0.1723757572f) {
                return -0.3109336759f;
            } else {
                return -0.5063160827f;
            }
        }
    } else {
        if (x[8] <= 0.4469437301f) {
            if (x[7] <= 0.1811790615f) {
                return 0.0505644743f;
            } else {
                return -0.0809695916f;
            }
        } else {
            if (x[8] <= 1.3758074045f) {
                return 0.0935668858f;
            } else {
                return 0.2060613511f;
            }
        }
    }
}
static float gbm_tree_23(const float *x) {
    if (x[0] <= -0.3245345652f) {
        if (x[3] <= -0.0801516101f) {
            if (x[8] <= -1.1073859334f) {
                return -0.1573281149f;
            } else {
                return -0.0330293440f;
            }
        } else {
            if (x[8] <= 0.0232480541f) {
                return -0.2449822134f;
            } else {
                return -0.4497614047f;
            }
        }
    } else {
        if (x[8] <= 0.9657530487f) {
            if (x[7] <= 0.2097332776f) {
                return 0.0862047639f;
            } else {
                return -0.0012557050f;
            }
        } else {
            if (x[8] <= 1.5357738137f) {
                return 0.1181872691f;
            } else {
                return 0.2466114118f;
            }
        }
    }
}
static float gbm_tree_24(const float *x) {
    if (x[0] <= -0.4139396697f) {
        if (x[3] <= 0.4711398631f) {
            if (x[8] <= -1.1524187922f) {
                return -0.1548999807f;
            } else {
                return -0.0408248411f;
            }
        } else {
            if (x[0] <= -1.7601438165f) {
                return -0.4999341984f;
            } else {
                return -0.2532326697f;
            }
        }
    } else {
        if (x[8] <= 0.8895484805f) {
            if (x[7] <= 0.0118761817f) {
                return 0.0760059049f;
            } else {
                return -0.0056763007f;
            }
        } else {
            if (x[8] <= 1.5553240180f) {
                return 0.1035359376f;
            } else {
                return 0.2478986405f;
            }
        }
    }
}
static float gbm_tree_25(const float *x) {
    if (x[0] <= 0.2041504160f) {
        if (x[0] <= -1.4809423685f) {
            if (x[0] <= -1.7147490382f) {
                return -0.3848726763f;
            } else {
                return -0.1925474354f;
            }
        } else {
            if (x[7] <= -1.5044462085f) {
                return 0.0641446726f;
            } else {
                return -0.0585941879f;
            }
        }
    } else {
        if (x[8] <= 0.0571555411f) {
            if (x[4] <= -0.4177714586f) {
                return 0.1791398011f;
            } else {
                return 0.6518635619f;
            }
        } else {
            if (x[8] <= 0.3740562946f) {
                return -0.0874857042f;
            } else {
                return 0.0789566668f;
            }
        }
    }
}
static float gbm_tree_26(const float *x) {
    if (x[0] <= -0.7792863548f) {
        if (x[0] <= -1.6166045070f) {
            if (x[5] <= 2.3645666838f) {
                return -0.2681810754f;
            } else {
                return -0.5835511596f;
            }
        } else {
            if (x[1] <= -0.6434578598f) {
                return -0.0778038661f;
            } else {
                return -0.3212249170f;
            }
        }
    } else {
        if (x[8] <= 0.7582862377f) {
            if (x[6] <= -0.7291767299f) {
                return 0.0582953554f;
            } else {
                return -0.0123879175f;
            }
        } else {
            if (x[8] <= 1.4712534547f) {
                return 0.0755158926f;
            } else {
                return 0.1766952706f;
            }
        }
    }
}
static float gbm_tree_27(const float *x) {
    if (x[0] <= -0.7382613122f) {
        if (x[9] <= 0.2188621908f) {
            if (x[8] <= -1.5470638871f) {
                return -0.1495179099f;
            } else {
                return -0.0523982865f;
            }
        } else {
            if (x[8] <= -0.5272734612f) {
                return -0.1983676942f;
            } else {
                return -0.3362513734f;
            }
        }
    } else {
        if (x[0] <= 0.2751176506f) {
            if (x[7] <= -0.7970902324f) {
                return 0.0550341525f;
            } else {
                return -0.0328582469f;
            }
        } else {
            if (x[8] <= 0.0522384113f) {
                return 0.3488976703f;
            } else {
                return 0.0580966012f;
            }
        }
    }
}
static float gbm_tree_28(const float *x) {
    if (x[0] <= -0.3434055448f) {
        if (x[2] <= -0.1151441187f) {
            if (x[8] <= -1.3163812160f) {
                return -0.1249490368f;
            } else {
                return -0.0240351414f;
            }
        } else {
            if (x[1] <= -0.1293940246f) {
                return -0.1577020915f;
            } else {
                return -0.3167313905f;
            }
        }
    } else {
        if (x[7] <= 0.2278590202f) {
            if (x[0] <= 0.9994342923f) {
                return 0.0598487585f;
            } else {
                return 0.1475002382f;
            }
        } else {
            if (x[8] <= 0.3506012112f) {
                return -0.0792777028f;
            } else {
                return 0.0408384820f;
            }
        }
    }
}
static float gbm_tree_29(const float *x) {
    if (x[0] <= 0.2437369153f) {
        if (x[3] <= 0.4928066880f) {
            if (x[0] <= -1.0707771182f) {
                return -0.0869034193f;
            } else {
                return -0.0122710426f;
            }
        } else {
            if (x[0] <= -1.7574562430f) {
                return -0.3372691654f;
            } else {
                return -0.1647118135f;
            }
        }
    } else {
        if (x[8] <= 0.0505349645f) {
            if (x[0] <= 1.0650457144f) {
                return 0.1484352573f;
            } else {
                return 0.6251527038f;
            }
        } else {
            if (x[8] <= 1.1001962423f) {
                return 0.0242562977f;
            } else {
                return 0.0863662031f;
            }
        }
    }
}
static float gbm_tree_30(const float *x) {
    if (x[0] <= -0.8119122386f) {
        if (x[0] <= -1.6937512159f) {
            if (x[5] <= 2.3374125957f) {
                return -0.2145140525f;
            } else {
                return -0.4348137235f;
            }
        } else {
            if (x[1] <= -0.7181142271f) {
                return -0.0583373547f;
            } else {
                return -0.2311316940f;
            }
        }
    } else {
        if (x[8] <= -0.1932622567f) {
            if (x[0] <= 0.4317931235f) {
                return -0.0250313404f;
            } else {
                return 0.1909465613f;
            }
        } else {
            if (x[7] <= 0.2617530823f) {
                return 0.0604694356f;
            } else {
                return 0.0087183352f;
            }
        }
    }
}
static float gbm_tree_31(const float *x) {
    if (x[0] <= -0.9007489085f) {
        if (x[0] <= -1.5321375728f) {
            if (x[5] <= 2.1751011610f) {
                return -0.1431829314f;
            } else {
                return -0.3107807440f;
            }
        } else {
            if (x[1] <= -0.7305543721f) {
                return -0.0496020752f;
            } else {
                return -0.2076850045f;
            }
        }
    } else {
        if (x[0] <= 0.2982152551f) {
            if (x[4] <= -0.5767415166f) {
                return 0.0325782759f;
            } else {
                return -0.0323521004f;
            }
        } else {
            if (x[8] <= 0.0522384113f) {
                return 0.2650465131f;
            } else {
                return 0.0404937137f;
            }
        }
    }
}
static float gbm_tree_32(const float *x) {
    if (x[0] <= -0.8439580798f) {
        if (x[5] <= -0.2462103367f) {
            if (x[10] <= -0.3535180986f) {
                return -0.0450172398f;
            } else {
                return -0.0125032506f;
            }
        } else {
            if (x[0] <= -1.7574833035f) {
                return -0.2402942883f;
            } else {
                return -0.0818472514f;
            }
        }
    } else {
        if (x[8] <= 1.0313922763f) {
            if (x[7] <= 0.0141859520f) {
                return 0.0261128634f;
            } else {
                return -0.0195110416f;
            }
        } else {
            if (x[8] <= 1.5739941001f) {
                return 0.0578753691f;
            } else {
                return 0.1697385127f;
            }
        }
    }
}
static float gbm_tree_33(const float *x) {
    if (x[0] <= 0.2975694686f) {
        if (x[7] <= 0.1787306070f) {
            if (x[7] <= -1.8027475476f) {
                return 0.1154749348f;
            } else {
                return -0.0097541196f;
            }
        } else {
            if (x[1] <= -0.0367019717f) {
                return -0.0529688810f;
            } else {
                return -0.2338941419f;
            }
        }
    } else {
        if (x[8] <= 0.0522384113f) {
            if (x[0] <= 0.7511959374f) {
                return 0.0874680002f;
            } else {
                return 0.4410035309f;
            }
        } else {
            if (x[8] <= 0.3383326828f) {
                return -0.1010992800f;
            } else {
                return 0.0383735668f;
            }
        }
    }
}
static float gbm_tree_34(const float *x) {
    if (x[0] <= -0.5356651843f) {
        if (x[1] <= -0.1751561314f) {
            if (x[8] <= -1.3364733458f) {
                return -0.0825649323f;
            } else {
                return -0.0179178532f;
            }
        } else {
            if (x[0] <= -1.2222901583f) {
                return -0.1496291894f;
            } else {
                return -0.2543604842f;
            }
        }
    } else {
        if (x[8] <= 1.2590202093f) {
            if (x[6] <= -0.0283624744f) {
                return 0.0381550068f;
            } else {
                return -0.0072346456f;
            }
        } else {
            if (x[8] <= 1.5990815759f) {
                return 0.0645141327f;
            } else {
                return 0.1749830260f;
            }
        }
    }
}
static float gbm_tree_35(const float *x) {
    if (x[8] <= -0.2934547961f) {
        if (x[10] <= -0.3535180986f) {
            if (x[8] <= -1.5103055835f) {
                return -0.1384260047f;
            } else {
                return -0.0523310569f;
            }
        } else {
            if (x[0] <= -1.6247832179f) {
                return -0.1099096612f;
            } else {
                return -0.0089453665f;
            }
        }
    } else {
        if (x[0] <= -0.7423332632f) {
            if (x[8] <= 0.3679360598f) {
                return -0.2157338471f;
            } else {
                return -0.0672784944f;
            }
        } else {
            if (x[8] <= 1.3758074045f) {
                return 0.0194088826f;
            } else {
                return 0.0822577211f;
            }
        }
    }
}
static float gbm_tree_36(const float *x) {
    if (x[0] <= 0.2982152551f) {
        if (x[7] <= -0.4284201860f) {
            if (x[7] <= -1.8626561165f) {
                return 0.1233991472f;
            } else {
                return 0.0059989333f;
            }
        } else {
            if (x[10] <= -0.3535180986f) {
                return -0.0845916515f;
            } else {
                return -0.0162767341f;
            }
        }
    } else {
        if (x[8] <= 0.0522384113f) {
            if (x[5] <= -0.1902478039f) {
                return 0.1273404294f;
            } else {
                return 0.4868011963f;
            }
        } else {
            if (x[8] <= 0.3824298829f) {
                return -0.0677522301f;
            } else {
                return 0.0308414049f;
            }
        }
    }
}
static float gbm_tree_37(const float *x) {
    if (x[0] <= -0.8702730238f) {
        if (x[0] <= -1.6364693046f) {
            if (x[6] <= 2.4238818884f) {
                return -0.1051752911f;
            } else {
                return -0.2785259723f;
            }
        } else {
            if (x[10] <= 0.3535180986f) {
                return -0.0529022951f;
            } else {
                return -0.0083120617f;
            }
        }
    } else {
        if (x[8] <= 1.0739059448f) {
            if (x[6] <= -0.5102886856f) {
                return 0.0271521455f;
            } else {
                return -0.0095256058f;
            }
        } else {
            if (x[8] <= 1.5800537467f) {
                return 0.0394279841f;
            } else {
                return 0.1288066852f;
            }
        }
    }
}
static float gbm_tree_38(const float *x) {
    if (x[0] <= 0.3555938303f) {
        if (x[1] <= 0.3433895856f) {
            if (x[10] <= 0.3535180986f) {
                return -0.0305088295f;
            } else {
                return 0.0121656031f;
            }
        } else {
            if (x[0] <= -1.2199313045f) {
                return -0.0826134926f;
            } else {
                return -0.2218920699f;
            }
        }
    } else {
        if (x[8] <= 0.0522384113f) {
            if (x[5] <= -0.1902478039f) {
                return 0.1257608659f;
            } else {
                return 0.4400427327f;
            }
        } else {
            if (x[8] <= 0.3118413538f) {
                return -0.0987110366f;
            } else {
                return 0.0244007327f;
            }
        }
    }
}
static float gbm_tree_39(const float *x) {
    if (x[8] <= 0.6070837975f) {
        if (x[7] <= 0.0666970462f) {
            if (x[0] <= 0.5738590658f) {
                return -0.0018109481f;
            } else {
                return 0.2338072447f;
            }
        } else {
            if (x[10] <= -0.3535180986f) {
                return -0.0671676920f;
            } else {
                return -0.0175186547f;
            }
        }
    } else {
        if (x[8] <= 1.4547224045f) {
            if (x[7] <= -0.3224160224f) {
                return 0.0375786168f;
            } else {
                return 0.0068062084f;
            }
        } else {
            if (x[8] <= 1.6364279389f) {
                return 0.0645553294f;
            } else {
                return 0.1584296747f;
            }
        }
    }
}
static float gbm_tree_40(const float *x) {
    if (x[0] <= -0.9517121017f) {
        if (x[0] <= -1.6825407743f) {
            if (x[5] <= 2.4238818884f) {
                return -0.1097556484f;
            } else {
                return -0.3122473849f;
            }
        } else {
            if (x[10] <= -0.3535180986f) {
                return -0.0597485170f;
            } else {
                return -0.0180004137f;
            }
        }
    } else {
        if (x[8] <= 1.0808675885f) {
            if (x[7] <= -0.7977312803f) {
                return 0.0265431967f;
            } else {
                return -0.0067301059f;
            }
        } else {
            if (x[8] <= 1.6150568128f) {
                return 0.0320072491f;
            } else {
                return 0.1302155967f;
            }
        }
    }
}
static float gbm_tree_41(const float *x) {
    if (x[0] <= -0.9325588644f) {
        if (x[0] <= -1.7212765217f) {
            if (x[6] <= 2.4728649855f) {
                return -0.1085710857f;
            } else {
                return -0.2546302173f;
            }
        } else {
            if (x[10] <= -1.0605542958f) {
                return -0.0699642827f;
            } else {
                return -0.0197684082f;
            }
        }
    } else {
        if (x[7] <= 0.2617530823f) {
            if (x[10] <= -0.3535180986f) {
                return -0.0007932054f;
            } else {
                return 0.0331451990f;
            }
        } else {
            if (x[8] <= 0.4093242586f) {
                return -0.0363998676f;
            } else {
                return 0.0099420149f;
            }
        }
    }
}
static float gbm_tree_42(const float *x) {
    if (x[0] <= -0.9517121017f) {
        if (x[0] <= -1.6454601288f) {
            if (x[4] <= 2.2880470753f) {
                return -0.0880278866f;
            } else {
                return -0.2939218531f;
            }
        } else {
            if (x[4] <= -0.7425868213f) {
                return 0.0026237850f;
            } else {
                return -0.0342355570f;
            }
        }
    } else {
        if (x[8] <= 1.2590202093f) {
            if (x[7] <= -0.5985031128f) {
                return 0.0218001005f;
            } else {
                return -0.0056572811f;
            }
        } else {
            if (x[8] <= 1.6421090364f) {
                return 0.0363240614f;
            } else {
                return 0.1278040966f;
            }
        }
    }
}
static float gbm_tree_43(const float *x) {
    if (x[8] <= -0.3309242129f) {
        if (x[10] <= 0.3535180986f) {
            if (x[1] <= -1.5116695762f) {
                return -0.0934336717f;
            } else {
                return -0.0257765466f;
            }
        } else {
            if (x[0] <= -0.1524568275f) {
                return 0.0009852811f;
            } else {
                return 0.1643681269f;
            }
        }
    } else {
        if (x[7] <= 0.2617530823f) {
            if (x[7] <= -1.8527654409f) {
                return 0.1241455341f;
            } else {
                return 0.0204056505f;
            }
        } else {
            if (x[9] <= 0.4054938108f) {
                return -0.0626146925f;
            } else {
                return 0.0070168503f;
            }
        }
    }
}
static float gbm_tree_44(const float *x) {
    if (x[0] <= -0.4861095399f) {
        if (x[1] <= -0.1751561314f) {
            if (x[10] <= 0.3535180986f) {
                return -0.0276367988f;
            } else {
                return 0.0059489877f;
            }
        } else {
            if (x[0] <= -0.6239256263f) {
                return -0.1084298596f;
            } else {
                return -0.2835779177f;
            }
        }
    } else {
        if (x[6] <= -1.8288081884f) {
            if (x[10] <= -1.0605542958f) {
                return 0.0663355351f;
            } else {
                return 0.1784363292f;
            }
        } else {
            if (x[8] <= 1.4232432842f) {
                return 0.0058800116f;
            } else {
                return 0.0510345044f;
            }
        }
    }
}
static float gbm_tree_45(const float *x) {
    if (x[0] <= 0.6080356240f) {
        if (x[4] <= -0.6657987237f) {
            if (x[7] <= -1.8074971437f) {
                return 0.0645607671f;
            } else {
                return 0.0083329962f;
            }
        } else {
            if (x[10] <= -1.0605542958f) {
                return -0.0711539156f;
            } else {
                return -0.0124927308f;
            }
        }
    } else {
        if (x[8] <= 0.0839584405f) {
            if (x[4] <= -0.0872514732f) {
                return 0.1407682105f;
            } else {
                return 0.3633603154f;
            }
        } else {
            if (x[8] <= 1.5667196512f) {
                return 0.0119178013f;
            } else {
                return 0.0794071971f;
            }
        }
    }
}
static float gbm_tree_46(const float *x) {
    if (x[8] <= -0.3057454377f) {
        if (x[10] <= -1.0605542958f) {
            if (x[1] <= -1.4536529183f) {
                return -0.1036717129f;
            } else {
                return -0.0381031672f;
            }
        } else {
            if (x[1] <= -1.5295931101f) {
                return -0.0461515755f;
            } else {
                return -0.0049648035f;
            }
        }
    } else {
        if (x[3] <= -0.1113273874f) {
            if (x[0] <= 0.8486405313f) {
                return 0.0247477891f;
            } else {
                return 0.1450094565f;
            }
        } else {
            if (x[9] <= 0.3405081928f) {
                return -0.0920780161f;
            } else {
                return 0.0082128608f;
            }
        }
    }
}
static float gbm_tree_47(const float *x) {
    if (x[0] <= 0.4160505533f) {
        if (x[10] <= -0.3535180986f) {
            if (x[7] <= 0.1755124778f) {
                return -0.0116724479f;
            } else {
                return -0.0662577920f;
            }
        } else {
            if (x[2] <= 0.3785406351f) {
                return 0.0084506105f;
            } else {
                return -0.0642997865f;
            }
        }
    } else {
        if (x[8] <= 0.0522384113f) {
            if (x[5] <= -0.1612730622f) {
                return 0.0936881988f;
            } else {
                return 0.3436127269f;
            }
        } else {
            if (x[8] <= 0.2704908103f) {
                return -0.0999637971f;
            } else {
                return 0.0117593938f;
            }
        }
    }
}
static float gbm_tree_48(const float *x) {
    if (x[7] <= -0.4495439678f) {
        if (x[10] <= -1.0605542958f) {
            if (x[2] <= -0.3660739064f) {
                return -0.0213443096f;
            } else {
                return 0.0052042889f;
            }
        } else {
            if (x[7] <= -1.9096052647f) {
                return 0.1353549859f;
            } else {
                return 0.0204878258f;
            }
        }
    } else {
        if (x[0] <= 0.7566367984f) {
            if (x[10] <= 0.3535180986f) {
                return -0.0313751105f;
            } else {
                return 0.0011930680f;
            }
        } else {
            if (x[8] <= 0.1302107209f) {
                return 0.2128911219f;
            } else {
                return 0.0073970965f;
            }
        }
    }
}
static float gbm_tree_49(const float *x) {
    if (x[0] <= -1.5756684542f) {
        if (x[7] <= 2.3209545612f) {
            if (x[10] <= 0.3535180986f) {
                return -0.0667413315f;
            } else {
                return -0.0158729507f;
            }
        } else {
            if (x[2] <= 1.6399452686f) {
                return -0.0934327528f;
            } else {
                return -0.1856195827f;
            }
        }
    } else {
        if (x[10] <= 1.0605542958f) {
            if (x[0] <= 0.3473261893f) {
                return -0.0115193243f;
            } else {
                return 0.0100976681f;
            }
        } else {
            if (x[4] <= 1.1799239516f) {
                return 0.0244954153f;
            } else {
                return -0.0161326915f;
            }
        }
    }
}
void NARX_narx_gbm_Init(NARX_narx_gbm_State_t *s){memset(s,0,sizeof(*s));}
float NARX_narx_gbm_Step(NARX_narx_gbm_State_t *s, float ac, float spd, float sa){
    float x[11]; int i;
    float ac_n=(ac-ROM_NARX_GBM_AC_MEAN)/ROM_NARX_GBM_AC_STD;
    float spd_n=(spd-ROM_NARX_GBM_SPD_MEAN)/ROM_NARX_GBM_SPD_STD;
    float sa_n=(sa-ROM_NARX_GBM_SA_MEAN)/ROM_NARX_GBM_SA_STD;
    x[0]=ac_n;
    for(i=0;i<3;i++) x[1+i]=s->ac_lag[i];
    x[4]=spd_n;
    for(i=0;i<3;i++) x[5+i]=s->spd_lag[i];
    x[8]=s->tq_lag[0]; x[9]=s->tq_lag[1];
    x[10]=sa_n;
    float y=0.00166431f;
    y+=0.100000f*gbm_tree_0(x);
    y+=0.100000f*gbm_tree_1(x);
    y+=0.100000f*gbm_tree_2(x);
    y+=0.100000f*gbm_tree_3(x);
    y+=0.100000f*gbm_tree_4(x);
    y+=0.100000f*gbm_tree_5(x);
    y+=0.100000f*gbm_tree_6(x);
    y+=0.100000f*gbm_tree_7(x);
    y+=0.100000f*gbm_tree_8(x);
    y+=0.100000f*gbm_tree_9(x);
    y+=0.100000f*gbm_tree_10(x);
    y+=0.100000f*gbm_tree_11(x);
    y+=0.100000f*gbm_tree_12(x);
    y+=0.100000f*gbm_tree_13(x);
    y+=0.100000f*gbm_tree_14(x);
    y+=0.100000f*gbm_tree_15(x);
    y+=0.100000f*gbm_tree_16(x);
    y+=0.100000f*gbm_tree_17(x);
    y+=0.100000f*gbm_tree_18(x);
    y+=0.100000f*gbm_tree_19(x);
    y+=0.100000f*gbm_tree_20(x);
    y+=0.100000f*gbm_tree_21(x);
    y+=0.100000f*gbm_tree_22(x);
    y+=0.100000f*gbm_tree_23(x);
    y+=0.100000f*gbm_tree_24(x);
    y+=0.100000f*gbm_tree_25(x);
    y+=0.100000f*gbm_tree_26(x);
    y+=0.100000f*gbm_tree_27(x);
    y+=0.100000f*gbm_tree_28(x);
    y+=0.100000f*gbm_tree_29(x);
    y+=0.100000f*gbm_tree_30(x);
    y+=0.100000f*gbm_tree_31(x);
    y+=0.100000f*gbm_tree_32(x);
    y+=0.100000f*gbm_tree_33(x);
    y+=0.100000f*gbm_tree_34(x);
    y+=0.100000f*gbm_tree_35(x);
    y+=0.100000f*gbm_tree_36(x);
    y+=0.100000f*gbm_tree_37(x);
    y+=0.100000f*gbm_tree_38(x);
    y+=0.100000f*gbm_tree_39(x);
    y+=0.100000f*gbm_tree_40(x);
    y+=0.100000f*gbm_tree_41(x);
    y+=0.100000f*gbm_tree_42(x);
    y+=0.100000f*gbm_tree_43(x);
    y+=0.100000f*gbm_tree_44(x);
    y+=0.100000f*gbm_tree_45(x);
    y+=0.100000f*gbm_tree_46(x);
    y+=0.100000f*gbm_tree_47(x);
    y+=0.100000f*gbm_tree_48(x);
    y+=0.100000f*gbm_tree_49(x);
    for(i=3;i>0;i--) s->ac_lag[i]=s->ac_lag[i-1]; s->ac_lag[0]=ac_n;
    for(i=3;i>0;i--) s->spd_lag[i]=s->spd_lag[i-1]; s->spd_lag[0]=spd_n;
    s->tq_lag[1]=s->tq_lag[0]; s->tq_lag[0]=y;
    return y*ROM_NARX_GBM_TQ_STD+ROM_NARX_GBM_TQ_MEAN;
}