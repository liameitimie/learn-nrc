target("layers")
    set_kind("static")
    add_files("*.cpp")
    add_deps("lc-dsl")
    add_deps("global", "activations", "gpu_rands", "optimizers")
    add_headerfiles("*.h")
    add_includedirs(".", {public=true})
target_end()