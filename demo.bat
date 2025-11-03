@REM python -m cast.cli -i assets/inputs/indoor.png --output output --enable-generation --pose-estimation-backend pytorch --generation-provider replicate --mesh-provider trellis --debug

@REM python -m cast.cli -i assets/inputs/image1.png --output output --enable-generation --pose-estimation-backend icp --generation-provider replicate --mesh-provider trellis
@REM python -m cast.cli -i assets/inputs/image2.png --output output --enable-generation --pose-estimation-backend icp --generation-provider replicate --mesh-provider trellis
@REM python -m cast.cli -i assets/inputs/image3.png --output output --enable-generation --pose-estimation-backend icp --generation-provider replicate --mesh-provider trellis

@REM python -m cast.cli -i assets/inputs/starbuck.png --output output --enable-generation --pose-estimation-backend pytorch ^
    @REM --generation-provider replicate --mesh-provider trellis --debug

@REM python -m cast.cli -i assets/inputs/car.png --output output --enable-generation --pose-estimation-backend pytorch ^
    @REM --generation-provider replicate --mesh-provider trellis --debug

@REM python -m cast.cli -i assets/inputs/book2.png --output output --enable-generation --pose-estimation-backend pytorch ^
@REM     --generation-provider qwen --mesh-provider trellis

@REM python -m cast.cli -i assets/inputs/doll2.png --output output --enable-generation --pose-estimation-backend pytorch ^
    @REM --generation-provider qwen --mesh-provider tripo3d

python -m cast.cli -i assets/inputs/car2.png --output output --enable-generation --pose-estimation-backend pytorch ^
    --generation-provider qwen --mesh-provider trellis --render-and-compare

@REM python -m cast.cli -i assets/inputs/bicycle2.png --output output --enable-generation --pose-estimation-backend pytorch ^
@REM     --generation-provider qwen --mesh-provider trellis

@REM python -m cast.cli -i assets/inputs/indoor.png --output output --enable-generation --pose-estimation-backend pytorch ^
    @REM --generation-provider qwen --mesh-provider trellis 