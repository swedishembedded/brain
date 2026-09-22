<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 52. Vulkan validation layers are one `apt-get download` away, even with no root and none installed

Two sessions' worth of "GPU device lost" on `backend-vulkan` was diagnosed in a
single run, by the layer, naming the exact VUID
(`VUID-vkCmdDispatch-None-08114`, a descriptor set using a destroyed buffer).
The blocker had looked structural: `/usr/share/vulkan/explicit_layer.d` holds
only Intel and Mesa layers on this box, and there is no root to install more.
Neither fact matters.

```
apt-get download vulkan-validationlayers   # no root needed to DOWNLOAD
dpkg-deb -x vulkan-validationlayers_*.deb vvl
VK_ADD_LAYER_PATH=vvl/usr/share/vulkan/explicit_layer.d \
LD_LIBRARY_PATH=vvl/usr/lib/x86_64-linux-gnu \
VK_LOADER_LAYERS_ENABLE='*validation*'  <any binary>
```

`VK_LOADER_LAYERS_ENABLE` (loader 1.3.234+; this box has 1.4.304) force-enables
a layer from outside the process, so **no debug path has to be added to any
`VkInstance` creation site** to get a diagnosis - which matters when the
instance is created in a crate you were told not to modify, or in a dependency.

The general shape: before reading synchronisation code cold, check whether the
API you are debugging has a validation/sanitizer mode and whether it can be
switched on from the environment. "The layer is not installed" is a statement
about the current filesystem, not about what is reachable.
