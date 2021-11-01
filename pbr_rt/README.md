Generate skybox:
```sh
_build/osx64_clang/bin/cmftRelease \
  --input ~/Downloads/11-13_Forest_D_scaled.hdr \
  --outputNum 1 \
  --output0 ~/Downloads/skybox \
  --output0params hdr,rgbe,facelist
```

Generate irradiance map:
```sh
_build/osx64_clang/bin/cmftRelease \
  --input ~/Downloads/11-13_Forest_D_scaled.hdr \
  --filter irradiance \
  --outputNum 1 \
  --output0 ~/Downloads/irrad \
  --output0params hdr,rgbe,facelist
```

Generate radiance map:
```sh
_build/osx64_clang/bin/cmftRelease \
  --input ~/Downloads/11-13_Forest_D_scaled.hdr \
  --filter radiance \
  --srcFaceSize 256 \
  --excludeBase false \
  --mipCount 6 \
  --glossScale 10 \
  --glossBias 1 \
  --lightingModel phongbrdf \
  --dstFaceSize 256 \
  --numCpuProcessingThreads 4 \
  --useOpenCL true \
  --clVendor anyGpuVendor \
  --deviceType gpu \
  --deviceIndex 0 \
  --inputGammaNumerator 1.0 \
  --inputGammaDenominator 1.0 \
  --outputGammaNumerator 1.0 \
  --outputGammaDenominator 1.0 \
  --generateMipChain false \
  --outputNum 2 \
  --output0 ~/Downloads/rad \
  --output0params hdr,rgbe,facelist
```
