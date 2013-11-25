1. INTRODUCTION
===============

  This project is an example of the Application Programmer's Interface
  for plugin-modules to the USP framework in 2300 scanner.

    . Stand-alone program/host, called "TheApplication". TheApplication
      functions basically as the USP host that loads the plug-in module.
      The calling sequence of the functions is the same.

    . A plugin DLL, called Plugin_A. Plugin_A is an illustration of how a 
      plugin module should implement the functions from the API.

    . UspPlugin.h - A header file that defines the interface functions.
      This header file is included both the by the host, TheApplication 
      and by the plugin - Plugin_A. 
      When the header file is included by the plugin, the following macro
      must be defined

            #define USP_PLUGIN_DLL   1

    . PyUspPlugin - a python class that is capable of loading the DLL and 
      calling the functions from the API. Similar functions/class can be 
      implemented in Matlab.

    This project has been successfully built and run on:

    . Windows 7, NVidia card, Visual Studio 10, ATI SDK
    . Windows 7, ATI card, Visual Studio 10, ATI SDK
    . Windows 7, ATI card, Visual Studio 12, ATI SDK
    . Mac OSX (Lion), No Card, standard installation of Xcode 4.2
    . Mac OSX (Mountain Lion), No card, standard installation of Xcode 
      (latest version)



2. NECESSARY TOOLS
==================

    . Compiler: gcc/g++ or MSVC will do. gcc 4.5 was used
      For Mac, get a copy of Xcode.

    . OpenCL libraries. ATI's and Apple's SDKs have been used
      Comes automatically on Mac with Xcode

    . Cmake, ver 2.8.10. Earlier versions will also do.
      http://cmake.org
      Cmake comes with GUI for Mac and Windows. It is easier to use the GUI.
    
    . Python environment (Optional)
        . Python ver 2.7.x or 2.6.x 
        . NumPy, SciPy, MatPlotLib, 
        . PyOpenCL

      A good distribution for Windows is Python(x,y):
         http://code.google.com/p/pythonxy/
      PyOpenCL can be found among the "Additional Plugins"
         http://code.google.com/p/pythonxy/wiki/AdditionalPlugins
    
      A good distribution for Mac is found from MacPorts. 
      To get the same environment as in Python(x,y) one needs to install
      Spyder, also available via Macports. Macports resolves dependencies
      automatically.


3. COMPILATION
==============
    . Open Cmake-gui
    . In source path select the directory SourceCode 
      (the directory where this README file resides)
    . In build path, select a directory where you want to build the project.
    . Run configure.
    . If CONFIGURE fails to detect OpenCL (happens in Windows), copy the file
      SourceCode/cmake/FindOpenCL  to the "Modules" subdirectory of CMake.
      The cause is a change in the environment name exported by AMD.
      Repeat Configure
    . Change the installation directory to somewhere you can find it.
    . Open your solution/project file
    . Build the project
    . Build the INSTALL target.
    . Start "TheApplication" from the ${INSTALLATION_DIR}/bin
    . You are done


    To test with Python you must edit the path specified on the following line:
    plugin = UspPlugin(r'c:\plugins\distro_win32\bin\plugins\plugin_a.dll')
    Run the script.

4. LICENSE
==========
  This software is distributed under the MIT license. 
  You can find it in the file License.txt in the same directory as this
  README file

