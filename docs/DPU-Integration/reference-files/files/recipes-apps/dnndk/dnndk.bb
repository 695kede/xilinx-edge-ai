#
# This file is the dnndk recipe.
#

SUMMARY = "Simple dnndk application"
SECTION = "PETALINUX/apps"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://bin/dexplorer \
           file://bin/dsight \
	   file://lib/echarts.js \
           file://lib/libdputils.so.3.1 \
           file://lib/libdsight.a \
           file://lib/libhineon.so \
           file://lib/libn2cube.so \
	   file://include/dnndk.h \
	   file://include/dputils.h \
	   file://include/n2cube.h \
	"
inherit module-base

S = "${WORKDIR}"

DEPENDS += "opencv"

TARGET_CC_ARCH += "${LDFLAGS}"

do_install() {
	     install -d ${D}/${bindir}
             install -m 0755 ${S}/bin/dexplorer ${D}/${bindir}
             install -m 0755 ${S}/bin/dsight ${D}/${bindir}

	     install -d ${D}${libdir}
             install -m 0655 ${S}/lib/echarts.js ${D}${libdir}
             install -m 0655 ${S}/lib/libdputils.so.3.1 ${D}${libdir}
             install -m 0655 ${S}/lib/libdsight.a ${D}${libdir}
             install -m 0655 ${S}/lib/libhineon.so ${D}${libdir}
             install -m 0655 ${S}/lib/libn2cube.so ${D}${libdir}
             cd ${D}${libdir}
             ln -s libdputils.so.3.1 libdputils.so

		
	     install -d ${D}/usr/local/lib
             cd ${D}/usr/local/lib
             ln -s ../../lib/libn2cube.so libn2cube.so 


             install -d ${D}/usr/include
             install -d ${D}/usr/include/dnndk
             install -m 0655 ${S}/include/dnndk.h ${D}${includedir}/dnndk/
             install -m 0655 ${S}/include/dputils.h ${D}${includedir}/dnndk/
             install -m 0655 ${S}/include/n2cube.h ${D}${includedir}/dnndk/
}


FILES_${PN} += "${libdir}"
FILES_${PN} += "/usr/local/lib"
FILES_${PN} += "/usr/local/include"
FILES_SOLIBSDEV = ""
INSANE_SKIP_${PN} += "dev-so"
