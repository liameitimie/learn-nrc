target("optimizers")
    set_kind("static")
    add_files("*.cpp")
    add_deps("lc-dsl")
    add_deps("global")
    add_headerfiles("*.h")
    add_includedirs(".", {public=true})
target_end()