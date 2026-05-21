# cmake/MacOSBundle.cmake
# Run after build to copy model into the app bundle and create DMG

set(APP_NAME "LLM Engine")
set(BUNDLE_PATH "${CMAKE_BINARY_DIR}/${APP_NAME}.app")
set(RESOURCES_DIR "${BUNDLE_PATH}/Contents/Resources")
set(MODEL_SRC "${CMAKE_SOURCE_DIR}/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
set(DMG_PATH "${CMAKE_BINARY_DIR}/LLM-Engine.dmg")

add_custom_target(copy_model
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RESOURCES_DIR}/models"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MODEL_SRC}"
        "${RESOURCES_DIR}/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    DEPENDS llm_desktop
    COMMENT "Copying model into app bundle..."
)

add_custom_target(create_dmg
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_BINARY_DIR}/dmg-staging"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/dmg-staging"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${BUNDLE_PATH}"
        "${CMAKE_BINARY_DIR}/dmg-staging/LLM Engine.app"
    COMMAND ln -sf /Applications "${CMAKE_BINARY_DIR}/dmg-staging/Applications"
    COMMAND hdiutil create
        -volname "LLM Engine"
        -srcfolder "${CMAKE_BINARY_DIR}/dmg-staging"
        -ov -format UDZO
        "${DMG_PATH}"
    DEPENDS copy_model
    COMMENT "Creating DMG..."
)
