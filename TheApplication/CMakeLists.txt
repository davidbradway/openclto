set (SRC  
     app_main.cpp 
	 )
	 

set (HDR
     ../UspPlugin/UspPlugin.h
     ../UspPlugin/UspDebug.h)

add_executable(TheApplication ${SRC} ${HDR})
add_dependencies(TheApplication "${PROJECT_SOURCE_DIR}/UspPlugin/UspPlugin.h")
target_link_libraries(TheApplication ${OPENCL_LIBRARIES})


if (MSVC)
   add_definitions( -D_CRT_SECURE_NO_WARNINGS  -W4 )    # Treat warnings as errors
endif(MSVC)

install(TARGETS TheApplication
        DESTINATION bin)
