# Local name for the module to avoid typos.
set(ModuleName ScoutMath)

# Defines needed by third party code.
add_compile_definitions(GLM_FORCE_RADIANS ON)
add_compile_definitions(GLM_FORCE_CXX17 ON)
add_compile_definitions(GLM_FORCE_EXPLICIT_CTOR ON) # Disallow implicit casts between types.
add_compile_definitions(GLM_FORCE_INLINE ON)
add_compile_definitions(GLM_FORCE_SIZE_T_LENGTH ON)

# Scout module's headers.
set(${ModuleName}_IncludePath ${PROJECT_SOURCE_DIR}/${ModuleName}/Include/)

# Scout module's sources.
file(GLOB ${ModuleName}_Inc ${PROJECT_SOURCE_DIR}/${ModuleName}/Include/Scout/*.h)
file(GLOB ${ModuleName}_Src ${PROJECT_SOURCE_DIR}/${ModuleName}/Src/*.cpp)

# Declaring the Scout module's static library target.
add_library(${ModuleName} STATIC ${${ModuleName}_Inc} ${${ModuleName}_Src})
set_target_properties(${ModuleName} PROPERTIES LINKER_LANGUAGE CXX) # Helps CMake out to figure out the language if it has trouble.

# Include paths for Scout module.
target_include_directories(${ModuleName} PUBLIC
	${${ModuleName}_IncludePath}
	${Glm_IncludePath}
)

# Linking the Scout module.
target_link_libraries(${ModuleName} PUBLIC
)
