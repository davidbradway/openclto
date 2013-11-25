set(HEADER
    ../UspPlugin/UspPlugin.h
    ../UspPlugin/UspDebug.h
	Parameters.h)


set(SRC  
    plugin_scale.cpp 
	../UspPlugin/UspDebug.cpp)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_library(plugin_b SHARED ${SRC} ${HEADER})
target_link_libraries(plugin_b ${OPENCL_LIBRARIES}  )

install(TARGETS plugin_b
        DESTINATION ${PLUGIN_INSTALL_DIR}
		)
install(FILES scale.cl
        DESTINATION ${PLUGIN_INSTALL_DIR})