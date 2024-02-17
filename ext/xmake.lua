includes("LuisaCompute")
includes("meshoptimizer")

-- package("meshoptimizer")
--     set_sourcedir(path.join(os.scriptdir(), "meshoptimizer"))

--     add_deps("cmake")
--     on_load("windows", function (package)
--         if package:config("shared") then
--             package:add("defines", "MESHOPTIMIZER_API=__declspec(dllimport)")
--         end
--     end)

--     on_install("windows", "macosx", "linux", function (package)
--         local configs = {}
--         table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
--         table.insert(configs, "-DMESHOPT_BUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
--         import("package.tools.cmake").install(package, configs)
--     end)

-- package_end()