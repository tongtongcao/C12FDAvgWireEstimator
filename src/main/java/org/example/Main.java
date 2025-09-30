package org.example;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslateException;
import ai.djl.inference.Predictor;
import ai.djl.translate.Batchifier;

import java.io.IOException;
import java.nio.file.Paths;

// Custom input class: contains a float array of length 5 and an integer
class MyInput {
    float[] data;   // 长度 = 5
    int maskIdx;    // 单个整数

    public MyInput(float[] data, int maskIdx) {
        this.data = data;
        this.maskIdx = maskIdx;
    }
}

public class Main {

    public static void main(String[] args) {

        // Define Translator: Input is MyInput, Output is Float
        Translator<MyInput, Float> myTranslator = new Translator<MyInput, Float>() {

            @Override
            public NDList processInput(TranslatorContext ctx, MyInput input) throws Exception {
                NDManager manager = ctx.getNDManager();

                // ---- 第一个输入：float 序列，形状 (1, 5, 1) ----
                NDArray x = manager.create(input.data).reshape(1, input.data.length, 1);

                // ---- 第二个输入：单个整数 mask_idx ----
                NDArray maskNd = manager.create(new int[]{input.maskIdx}); // shape (1,)

                return new NDList(x, maskNd);
            }

            @Override
            public Float processOutput(TranslatorContext ctx, NDList list) throws Exception {
                NDArray result = list.get(0);
                return result.getFloat();   // return single float
            }

            @Override
            public Batchifier getBatchifier() {
                return null;
            }
        };

        // Define Criteria
        Criteria<MyInput, Float> myModelCriteria = Criteria.builder()
                .setTypes(MyInput.class, Float.class)
                .optModelPath(Paths.get("nets/tmae_default.pt"))  // replace with your model path
                .optEngine("PyTorch")
                .optTranslator(myTranslator)
                .optProgress(new ProgressBar())
                .build();

        // Run inference
        try (ZooModel<MyInput, Float> model = myModelCriteria.loadModel();
             Predictor<MyInput, Float> predictor = model.newPredictor()) {

            // Example input: float array of length 5
            float[] dummyArray = new float[]{56.0000f,57.8333f,59.3333f,65.3333f,68.8000f};

            // integer parameter
            int myIntParam = 2;

            // wrap inputs
            MyInput input = new MyInput(dummyArray, myIntParam);

            // predict
            Float output = predictor.predict(input);
            System.out.println("Predicted value: " + output);

        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException(e);
        }
    }
}

