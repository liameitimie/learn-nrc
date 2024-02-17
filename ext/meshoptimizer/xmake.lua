target("meshoptimizer")
    set_kind("static")
    add_files("src/**.cpp")
    add_deps("lc-core")
    add_headerfiles("src/*.h")
    add_includedirs("src", {public = true})
    if is_kind("shared") and is_os("windows") then
        add_defines("MESHOPTIMIZER_API=__declspec(dllexport)")
        add_defines("MESHOPTIMIZER_API=__declspec(dllimport)", {public = true})
    end
target_end()