/*
 Copyright (c) 2010 Michal Kottman

 Permission is hereby granted, free of charge, to any person
 obtaining a copy of this software and associated documentation
 files (the "Software"), to deal in the Software without
 restriction, including without limitation the rights to use,
 copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following
 conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
*/

#include <stdlib.h>

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#define CV_META_IMAGE   "lcvImage"
#define CV_META_MAT     "lcvMat"
#define CV_META_CAPTURE "lcvCapture"
#define CV_META_CORNERS "lcvCorners"

#define NYI return luaL_error(L, "not yet implemented (%s)", __FUNCTION__)

#define CV_FUNC(f) static int lcv_cv_##f(lua_State *L)
#define HG_FUNC(f) static int lcv_hg_##f(lua_State *L)
#define IV_FUNC(f) static int lcv_iv_##f(lua_State *L)

/**********************************
******* Utility functions *********
***********************************/

#define LCV_FLAG_RELEASE    1

/* save space for pointer and flags */
#define LCV_NEW(TYPE, NAME, GC)     ((TYPE**) lcvL_newuserdata(L, NAME, sizeof(TYPE*), GC))
#define LCV_NEW_GC(TYPE, NAME)      ((TYPE**) lcvL_newuserdata(L, NAME, sizeof(TYPE*), 1))
#define LCV_NEW_NOGC(TYPE, NAME)    ((TYPE**) lcvL_newuserdata(L, NAME, sizeof(TYPE*), 0))
#define LCV_GET(TYPE, NAME, IDX)    ((TYPE**) luaL_checkudata(L, IDX, NAME))
#define LCV_SELF(TYPE, NAME)        LCV_GET(TYPE, NAME, 1)

#define LCV_FLAGS(ptr, size)        ((int*) (((char*) ptr) + size))

static void dumpstack(lua_State *L) {
    int i, n = lua_gettop(L);
    for (i=1; i<=n; i++) {
        printf("%03d: %10s ", i, luaL_typename(L, i));
        switch (lua_type(L, i)) {
            case LUA_TNIL: printf("nil\n"); break;
            case LUA_TBOOLEAN: printf("%s\n", lua_toboolean(L, i) ? "true" : "false"); break;
            case LUA_TSTRING: printf("%s\n", lua_tostring(L, i)); break;
            case LUA_TNUMBER: printf("%f\n", lua_tonumber(L, i)); break;
            default: printf("%p\n", lua_topointer(L, i)); break;
        }
    }
}

static CvArr * lcvL_toCvArr(lua_State *L, int idx) {
    switch (lua_type(L, idx)) {
        case LUA_TUSERDATA: {
                IplImage ** img = LCV_GET(IplImage, CV_META_IMAGE, idx);
                return *img;
            }
            break;
        case LUA_TTABLE:
            break;
        default:
            luaL_typerror(L, idx, "table/userdata");
    }
    return NULL;
}

static CvTermCriteria lcvL_toCriteria(lua_State *L, int idx) {
    CvTermCriteria crit = {0};
    lua_getfield(L, idx, "eps");
    if (!lua_isnil(L, -1)) {
        crit.type = CV_TERMCRIT_EPS;
        crit.epsilon = lua_tointeger(L, -1);
    }
    lua_getfield(L, idx, "iter");
    if (!lua_isnil(L, -1)) {
        crit.type |= CV_TERMCRIT_ITER;
        crit.max_iter = lua_tointeger(L, -1);
    }
    lua_pop(L, 2);
    return crit;
}

static CvSize lcvL_toCvSize(lua_State *L, int idx) {
    luaL_checktype(L, idx, LUA_TTABLE);
    lua_rawgeti(L, idx, 1);
    int x = lua_tointeger(L, -1);
    lua_rawgeti(L, idx, 2);
    int y = lua_tointeger(L, -1);
    lua_pop(L, 2);
    return cvSize(x, y);
}

static int lcvL_shouldRelease(void * ptr, int size) {
    int * flags = LCV_FLAGS(ptr, size);
    return * flags & LCV_FLAG_RELEASE;
}

static void * lcvL_newuserdata(lua_State *L, const char * metaname, size_t size, int should_gc) {
    void * ret = lua_newuserdata(L, size + sizeof(int));
    int * flags = LCV_FLAGS(ret, size);
    *flags = should_gc;
    luaL_getmetatable(L, metaname);
    lua_setmetatable(L, -2);
    return ret;
}

static int lcvL_pushsize(lua_State *L, CvSize size) {
    lua_createtable(L, 2, 0);
    lua_pushnumber(L, size.width);
    lua_rawseti(L, -2, 1);
    lua_pushnumber(L, size.height);
    lua_rawseti(L, -2, 2);
    return 1;
}

static int lcvL_pushimage(lua_State *L, IplImage *img, int should_gc) {
    if (!img) {
        lua_pushnil(L);
    } else {
        IplImage ** ret = LCV_NEW(IplImage, CV_META_IMAGE, should_gc);
        *ret = img;
    }
    return 1;
}

/**********************************
******** Cv functions ********
***********************************/

static int IMG_depth_values [] = {
        IPL_DEPTH_8U,
        IPL_DEPTH_8S,
        IPL_DEPTH_16U,
        IPL_DEPTH_16S,
        IPL_DEPTH_32S,
        IPL_DEPTH_32F,
        IPL_DEPTH_64F
};

static const char * IMG_depth_names[] = {
        "8u", "8s", "16u", "16s", "32s", "32f", "64f"
};

CV_FUNC(CreateImage) {
    CvSize size = lcvL_toCvSize(L, 1);
    int depthIdx = luaL_checkoption(L, 2, "8u", IMG_depth_names);
    int channels = luaL_optinteger(L, 3, 3);

    IplImage *img = cvCreateImage(size, IMG_depth_values[depthIdx], channels);
    return lcvL_pushimage(L, img, 1);
}

static const char * IMG_cvtcolor_names[] = {
        "rgb2gray",
        /* TODO: rest */
};

static int IMG_cvtcolor_values[] = {
        CV_RGB2GRAY,
        CV_BGR2XYZ, CV_RGB2XYZ, CV_XYZ2BGR, CV_XYZ2RGB,
        CV_BGR2YCrCb, CV_RGB2YCrCb, CV_YCrCb2BGR, CV_YCrCb2RGB,
        CV_BGR2HSV, CV_RGB2HSV, CV_HSV2BGR, CV_HSV2RGB,
        CV_BGR2HLS, CV_RGB2HLS, CV_HLS2BGR, CV_HLS2RGB,
        CV_BGR2Lab, CV_RGB2Lab, CV_Lab2BGR, CV_Lab2RGB,
        CV_BGR2Luv, CV_RGB2Luv, CV_Luv2BGR, CV_Luv2RGB,
};

CV_FUNC(CvtColor) {
    CvArr * src = lcvL_toCvArr(L, 1);
    CvArr * dst = lcvL_toCvArr(L, 2);
    int codeIdx = luaL_checkoption(L, 3, NULL, IMG_cvtcolor_names);
    cvCvtColor(src, dst, IMG_cvtcolor_values[codeIdx]);
    return 0;
}

CV_FUNC(GetSize) {
    CvArr * arr = lcvL_toCvArr(L, 1);
    CvSize size = cvGetSize(arr);
    return lcvL_pushsize(L, size);
}

/* Chessboard */

CV_FUNC(FindChessboardCorners) {
    CvArr *img = lcvL_toCvArr(L, 1);
    CvSize size = lcvL_toCvSize(L, 2);
    int count = size.width * size.height;

    CvPoint2D32f *corners = (CvPoint2D32f*) malloc(count * sizeof(CvPoint2D32f));

    int ret = cvFindChessboardCorners(img, size, corners, &count, CV_CALIB_CB_ADAPTIVE_THRESH);
    if (ret == 0) {
        lua_pushnil(L);
        return 1;
    } else {
        // return only the corners found
        CvPoint2D32f *result = (CvPoint2D32f*) lcvL_newuserdata(L, CV_META_CORNERS, count * sizeof(CvPoint2D32f), 1);
        memcpy(result, corners, count * sizeof(CvPoint2D32f));
        free(corners);
        lua_pushnumber(L, count);
        return 2;
    }
}

CV_FUNC(FindCornerSubPix) {
    CvArr *img = lcvL_toCvArr(L, 1);
    CvPoint2D32f *corners = luaL_checkudata(L, 2, CV_META_CORNERS);
    int count = lua_objlen(L, 2) / sizeof(CvPoint2D32f);
    int winsize = luaL_checkinteger(L, 3);
    CvTermCriteria criteria = lcvL_toCriteria(L, 4);

    cvFindCornerSubPix(img, corners, count, cvSize(winsize, winsize), cvSize(-1, -1), criteria);
    return 0;
}

CV_FUNC(DrawChessboardCorners) {
    CvArr *img = lcvL_toCvArr(L, 1);
    CvSize size = lcvL_toCvSize(L, 2);
    CvPoint2D32f *corners = luaL_checkudata(L, 3, CV_META_CORNERS);
    int count = lua_objlen(L, 3) / sizeof(CvPoint2D32f);

    cvDrawChessboardCorners(img, size, corners, count, 1);
    return 0;
}

/**********************************
**** Image and video functions ****
***********************************/

IV_FUNC(LoadImage) {
    const char * filename = luaL_checkstring(L, 1);
    int iscolor = luaL_optinteger(L, 2, 1);
    IplImage *img = cvLoadImage(filename, iscolor);
    return lcvL_pushimage(L, img, 1);
}

IV_FUNC(SaveImage) {
    const char * filename = luaL_checkstring(L, 1);
    CvArr * img = lcvL_toCvArr(L, 2);
    cvSaveImage(filename, img, 0);
    return 0;
}

IV_FUNC(CaptureFromCAM) {
    int index = luaL_optint(L, 1, -1);
    CvCapture ** cap = LCV_NEW_GC(CvCapture, CV_META_CAPTURE);
    *cap = cvCaptureFromCAM(index);
    return 1;
}

IV_FUNC(CaptureFromFile) {
    const char * filename = luaL_checkstring(L, 1);
    CvCapture ** cap = LCV_NEW_GC(CvCapture, CV_META_CAPTURE);
    *cap = cvCaptureFromFile(filename);
    return 1;
}

IV_FUNC(GrabFrame) {
    CvCapture **cap = LCV_SELF(CvCapture, CV_META_CAPTURE);
    cvGrabFrame(*cap);
    return 0;
}

IV_FUNC(QueryFrame) {
    CvCapture **cap = LCV_SELF(CvCapture, CV_META_CAPTURE);
    IplImage *img = cvQueryFrame(*cap);
    return lcvL_pushimage(L, img, 0);
}

IV_FUNC(RetrieveFrame) {
    CvCapture **cap = LCV_SELF(CvCapture, CV_META_CAPTURE);
    IplImage *img = cvRetrieveFrame(*cap, 0);
    return lcvL_pushimage(L, img, 0);
}

IV_FUNC(__gc) {
    CvCapture **cap = LCV_SELF(CvCapture, CV_META_CAPTURE);
    if (lcvL_shouldRelease(cap, sizeof(CvCapture*))) {
        cvReleaseCapture(cap);
    }
    return 0;
}

const int CAP_prop_values[] = {
        CV_CAP_PROP_POS_MSEC, CV_CAP_PROP_POS_FRAMES, CV_CAP_PROP_POS_AVI_RATIO,
        CV_CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_HEIGHT, CV_CAP_PROP_FPS,
        CV_CAP_PROP_FOURCC, CV_CAP_PROP_FRAME_COUNT, CV_CAP_PROP_BRIGHTNESS,
        CV_CAP_PROP_CONTRAST, CV_CAP_PROP_SATURATION, CV_CAP_PROP_HUE
};

const char * CAP_prop_names[] = {
        "pos_msec", "pos_frames", "pos_ratio",
        "width", "height", "fps",
        "fourcc", "frame_count", "brightness",
        "contrast", "saturation", "hue"
};

IV_FUNC(__index) {
    // first, check if it should be a function
    luaL_getmetatable(L, CV_META_CAPTURE);  // Tab Key Meta
    lua_pushvalue(L, 2);                    // Tab Key Meta Key
    lua_gettable(L, 3);                     // Tab Key Fun/nil
    if (!lua_isnil(L, -1)) {
        return 1;
    }
    // else, return a property value
    CvCapture **cap = LCV_SELF(CvCapture, CV_META_CAPTURE);
    const char * key = luaL_checkstring(L, 2);
    int idx = luaL_checkoption(L, 2, NULL, CAP_prop_names);
    double val = cvGetCaptureProperty(*cap, CAP_prop_values[idx]);
    lua_pushnumber(L, val);
    return 1;
}

IV_FUNC(__newindex) {
    CvCapture **cap = LCV_SELF(CvCapture, CV_META_CAPTURE);
    const char * key = luaL_checkstring(L, 2);
    double val = luaL_checknumber(L, 3);
    int idx = luaL_checkoption(L, 2, NULL, CAP_prop_names);
    cvSetCaptureProperty(*cap, CAP_prop_values[idx], val);
    return 0;
}

/**********************************
******* HighGUI functions *********
***********************************/

HG_FUNC(ConvertImage) {
    NYI;
}

/* workaround for the extremely simple CreateTrackbar API */

#define MAX_TRACKBARS 24
typedef struct { lua_State *L; int ref, val; } TrackbarEntry;
static TrackbarEntry trackbars[MAX_TRACKBARS];

static void track(int trackbar, int pos) {
    TrackbarEntry te = trackbars[trackbar-1];
    if (!te.L)
        return;
    lua_rawgeti(te.L, LUA_REGISTRYINDEX, te.ref);
    lua_pushinteger(te.L, pos);
    lua_call(te.L, 1, 0);
}

#define TRACKER(n) \
static void tracker##n(int pos) { track(n, pos); }

/* this has to be done manually */
TRACKER( 1) TRACKER( 2) TRACKER( 3) TRACKER( 4) TRACKER( 5) TRACKER( 6)
TRACKER( 7) TRACKER( 8) TRACKER( 9) TRACKER(10) TRACKER(11) TRACKER(12)
TRACKER(13) TRACKER(14) TRACKER(15) TRACKER(16) TRACKER(17) TRACKER(18)
TRACKER(19) TRACKER(20) TRACKER(21) TRACKER(22) TRACKER(23) TRACKER(24)

CvTrackbarCallback trackers[] = {
    tracker1 , tracker2 , tracker3 , tracker4 , tracker5 , tracker6,
    tracker7 , tracker8 , tracker9 , tracker10, tracker11, tracker12,
    tracker13, tracker14, tracker15, tracker16, tracker17, tracker18,
    tracker19, tracker20, tracker21, tracker22, tracker23, tracker24,
};

HG_FUNC(CreateTrackbar) {
    static int tb_count = 0;

    if (tb_count == MAX_TRACKBARS) {
        return luaL_error(L, "max number of trackbars (%d) reached", MAX_TRACKBARS);
    }

    const char * tbName = luaL_checkstring(L, 1);
    const char * wndName = luaL_checkstring(L, 2);
    int val = luaL_checkint(L, 3);
    int max = luaL_checkint(L, 4);
    luaL_checktype(L, 5, LUA_TFUNCTION);
    
    trackbars[tb_count].L = L;
    trackbars[tb_count].val = val;
    trackbars[tb_count].ref = luaL_ref(L, LUA_REGISTRYINDEX);
    cvCreateTrackbar(tbName, wndName, &trackbars[tb_count].val, max, trackers[tb_count]);  
    tb_count++;

    return 0;
}

HG_FUNC(DestroyAllWindows) {
    cvDestroyAllWindows();
    return 0;
}

HG_FUNC(DestroyWindow) {
	const char * name = luaL_checkstring(L, 1);
	cvDestroyWindow(name);
	return 0;
}

HG_FUNC(GetTrackbarPos) {
	const char * tbname = luaL_checkstring(L, 1);
	const char * wname = luaL_checkstring(L, 2);
	int val = cvGetTrackbarPos(tbname, wname);
    lua_pushinteger(L, val);
	return 1;
}

HG_FUNC(MoveWindow) {
    const char * name = luaL_checkstring(L, 1);
    int x = luaL_checkint(L, 2);
    int y = luaL_checkint(L, 3);
    cvMoveWindow(name, x, y);
    return 0;
}

HG_FUNC(NamedWindow) {
	const char * name = luaL_checkstring(L, 1);
	cvNamedWindow(name, CV_WINDOW_AUTOSIZE);
	return 0;
}

HG_FUNC(ResizeWindow) {
    const char * name = luaL_checkstring(L, 1);
    int width = luaL_checkint(L, 2);
    int height = luaL_checkint(L, 3);
    cvResizeWindow(name, width, height);
    return 0;   
}

HG_FUNC(SetMouseCallback) {
    NYI;
}

HG_FUNC(SetTrackbarPos) {
    const char * tname = luaL_checkstring(L, 1);
    const char * wname = luaL_checkstring(L, 2);
    int pos = luaL_checkint(L, 3);
    cvSetTrackbarPos(tname, wname, pos);
    return 0;
}

HG_FUNC(ShowImage) {
    const char * name = luaL_checkstring(L, 1);
    const CvArr * img = lcvL_toCvArr(L, 2);
    cvShowImage(name, img);
    return 0;
}

HG_FUNC(WaitKey) {
	int delay = luaL_optint(L, 1, 0);
	int key = cvWaitKey(delay);
    if (key == -1) {
        // no key
        lua_pushnil(L); return 1;
    } else {
        // push key and modifiers (GUESS)
        lua_pushinteger(L, key & 0xFFFF);
        lua_pushinteger(L, key >> 16);
        return 2;
    }
}

#define CV(n) {#n, lcv_cv_##n}
#define HG(n) {#n, lcv_hg_##n}
#define IV(n) {#n, lcv_iv_##n}

static luaL_Reg cv_functions[] = {
    /* Cv */
    CV(CreateImage),
    CV(CvtColor),
    CV(GetSize),
    CV(FindChessboardCorners),
    CV(FindCornerSubPix),
    CV(DrawChessboardCorners),
    /* Image/Video */
    IV(LoadImage),
    IV(SaveImage),
    IV(CaptureFromCAM),
    IV(CaptureFromFile),
    /* HighGUI */
    HG(CreateTrackbar),
    HG(DestroyAllWindows),
    HG(DestroyWindow),
    HG(GetTrackbarPos),
    HG(MoveWindow),
	HG(NamedWindow),
    HG(ResizeWindow),
    HG(SetMouseCallback),
    HG(SetTrackbarPos),
    HG(ShowImage),
    HG(WaitKey),
	{NULL, NULL}
};

static luaL_Reg cv_cap_methods[] = {
    IV(GrabFrame),
    IV(RetrieveFrame),
    IV(QueryFrame),
    IV(__gc),
    IV(__index),
    IV(__newindex),
    {NULL, NULL}
};

static luaL_Reg cv_empty_methods[] = {
    {NULL, NULL}
};

void lcvL_newmeta(lua_State *L, const char *metaname, const luaL_Reg *methods) {
    luaL_newmetatable(L, metaname);
    // define __index first to allow overrides
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");
    luaL_register(L, NULL, methods);
}

LUA_API int luaopen_luacv(lua_State *L) {
    lcvL_newmeta(L, CV_META_CAPTURE, cv_cap_methods);
    lcvL_newmeta(L, CV_META_IMAGE, cv_empty_methods);
    lcvL_newmeta(L, CV_META_CORNERS, cv_empty_methods);
    luaL_register(L, "luacv", cv_functions);
    return 1;
}
