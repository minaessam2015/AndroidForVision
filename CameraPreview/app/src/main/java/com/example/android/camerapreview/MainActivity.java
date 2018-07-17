package com.example.android.camerapreview;

import android.hardware.Camera;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;

import java.util.List;

public class MainActivity extends AppCompatActivity {
    private final String tag = "MainActivity";
    Camera camera = null;
    CameraPreview preview;
    Button button;
    FrameLayout frameLayout;
    boolean capture =true;
    List<Integer> formats;
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        frameLayout=(FrameLayout)findViewById(R.id.camera_preview);
        button = (Button) findViewById(R.id.capture_stop_button);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(capture){
                    button.setText("stop");
                    capture=false;
                    try{
                        camera= Camera.open(0);
                    }catch (Exception e){
                        Log.d(tag,"Error opening the camera "+ e.getMessage());
                    }


                    if(camera!=null){
                        formats = camera.getParameters().getSupportedPreviewFormats();
                        for(int i=0 ; i< formats.size();i++){
                            Log.d(tag,formats.get(i).toString());
                        }
                        List<Camera.Size> previewSizes=camera.getParameters().getSupportedPreviewSizes();
                        double minDiff=10000000;
                        Camera.Size minSize=previewSizes.get(0);
                        for (int i=0;i<previewSizes.size();i++){
                           // Log.d(tag, "width  "+String.valueOf(previewSizes.get(i).width)+"   height  "+String.valueOf(previewSizes.get(i).height));
                           double strideW=((previewSizes.get(i).width )/415.0);
                            double strideH = ((previewSizes.get(i).height)/415.0);
                            double diff=(strideW-Math.floor(strideW))+(strideH-Math.floor(strideH));
                            if(diff<minDiff){
                                minDiff=diff;
                                minSize=previewSizes.get(i);
                            }

                        }
                        Log.d(tag,"min preview size   w "+minSize.width+"   h  "+minSize.height);
                        Camera.Parameters parameters=camera.getParameters();
                        parameters.setPreviewSize(minSize.width,minSize.height);
                        camera.setParameters(parameters);
                        preview= new CameraPreview(getApplicationContext(),camera);
                        frameLayout.addView(preview);
                    }

                }else {
                    button.setText("capture");
                    capture=true;
                    if(camera != null){
                        camera.stopPreview();
                        camera.setPreviewCallback(null);
                        frameLayout.removeAllViews();
                        camera.release();
                        camera=null;
                        preview=null;
                    }
                }
            }
        });

    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}
