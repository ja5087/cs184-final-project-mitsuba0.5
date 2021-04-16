BUILDDIR       = '#build/release'
DISTDIR        = '#Mitsuba.app'
CXX            = 'clang++'
CC             = 'clang'
CCFLAGS        = ['-mmacosx-version-min=11.1', '-march=native', '-funsafe-math-optimizations', '-fno-math-errno', '-isysroot', '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.1.sdk', '-O3', '-Wall', '-Wno-deprecated-declarations', '-g', '-DMTS_DEBUG', '-DSINGLE_PRECISION', '-DSPECTRUM_SAMPLES=3', '-DMTS_SSE', '-DMTS_HAS_COHERENT_RT', '-fvisibility=hidden', '-ftemplate-depth=512', '-stdlib=libc++']
LINKFLAGS      = ['-framework', 'OpenGL', '-framework', 'Cocoa', '-mmacosx-version-min=11.1', '-Wl,-syslibroot,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.1.sdk', '-Wl,-headerpad,128', '-stdlib=libc++']
CXXFLAGS       = ['-std=c++11']
BASEINCLUDE    = ['#include', '#dependencies/include']
BASELIBDIR     = ['#dependencies/lib']
BASELIB        = ['m', 'pthread', 'Half']
OEXRINCLUDE    = ['#dependencies/include/OpenEXR']
OEXRLIB        = ['IlmImf', 'Imath', 'Iex', 'z']
PNGLIB         = ['png16']
PNGINCLUDE     = ['#dependencies/include/libpng']
JPEGLIB        = ['jpeg']
JPEGINCLUDE    = ['#dependencies/include/libjpeg']
XERCESLIB      = ['xerces-c']
GLLIB          = ['GLEWmx', 'objc']
GLFLAGS        = ['-DGLEW_MX']
BOOSTINCLUDE   = ['#dependencies']
BOOSTLIB       = ['boost_filesystem', 'boost_system', 'boost_thread']
PYTHON27INCLUDE= ['/System/Library/Frameworks/Python.framework/Versions/2.7/Headers']
PYTHON27LIBDIR = ['/System/Library/Frameworks/Python.framework/Versions/2.7/lib']
PYTHON27LIB    = ['boost_python27', 'boost_system']
PYTHON35INCLUDE= ['#dependencies/include/python3.5']
PYTHON35LIB    = ['boost_python36', 'boost_system']
PYTHON36INCLUDE= ['#dependencies/include/python3.4']
PYTHON36LIB    = ['boost_python36', 'boost_system']
# COLLADAINCLUDE = ['#dependencies/include/collada-dom', '#dependencies/include/collada-dom/1.4']
# COLLADALIB     = ['collada14dom24']
QTDIR          = '#dependencies'
FFTWLIB        = ['fftw3']
