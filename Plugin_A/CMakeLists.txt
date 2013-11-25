set(HEADER
    ../UspPlugin/UspPlugin.h
    ../UspPlugin/UspDebug.h)

set(SRC  
    plugin_main.cpp 
	../UspPlugin/UspDebug.cpp)
	 



add_library(plugin_a SHARED ${SRC} ${HEADER})
target_link_libraries(plugin_a ${OPENCL_LIBRARIES}  )





install(TARGETS plugin_a
        DESTINATION ${PLUGIN_INSTALL_DIR}
		)

	 

	 
	
