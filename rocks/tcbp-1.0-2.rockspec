package = "tcbp"
version = "1.0-2"

source = {
   url = "git://github.com/yuejiashen/CompactBilinearPooling",
   tag = "master"
 }

description = {
   summary = "Compact Bilinear Pooling for Torch7 nn",
   detailed = [[
   Torch7 Implementation of Compact Bilnear Pooling for image input.
   ]],
   homepage = "https://github.com/yuejiashen/CompactBilinearPooling",
   license = "BSD-3 Clause"
}

dependencies = {
   "nn >= 1.0",
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
   cmake -E make_directory build;
   cd build;
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
   $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
