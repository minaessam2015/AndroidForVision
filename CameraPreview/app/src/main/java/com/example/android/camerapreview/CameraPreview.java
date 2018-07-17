package com.example.android.camerapreview;

import android.content.Context;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;

/**
 * Created by mina essam on 12-Jul-18.
 */

public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {
    private final String tag="CameraPreview";
    private Camera camera;
    private SurfaceHolder surfaceHolder;


    public CameraPreview(Context context , Camera camera){
        super(context);
        this.camera=camera;
        surfaceHolder=getHolder();
        surfaceHolder.addCallback(this);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        try{
            camera.setPreviewDisplay(surfaceHolder);
            camera.startPreview();
        }catch (IOException e){
            Log.e(tag,"Error setting camera preview "+ e.getMessage());
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {

        if(surfaceHolder.getSurface()==null){
            return;
        }

        try {
            camera.stopPreview();
        }catch (Exception e){

        }

        try{
            camera.setPreviewDisplay(surfaceHolder);
            camera.startPreview();
        }catch (Exception e){
            Log.d(tag,"Error setting camera preview "+ e.getMessage());
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {

        camera.stopPreview();
        camera.release();
    }
}
