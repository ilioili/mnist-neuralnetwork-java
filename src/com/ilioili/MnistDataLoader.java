package com.ilioili;

import java.io.*;
import java.util.Arrays;

/**
 * Created by ilioili on 2018/1/4.
 * 关于数据格式介绍参考 http://yann.lecun.com/exdb/mnist/
 */
public class MnistDataLoader {
    public static byte[] imageBytePixels;
    public static double[] imageDoublePixels;
    public static byte[] label;
    public static final int LENGHT = 784;
    public static final double[] LABEL_ARRAY = new double[10];

    public static void loadData() {
        loadImageData();
        loadLabelData();
    }

    final static double[] imageData = new double[LENGHT];

    public static double[] getImageData(int index) {
        System.arraycopy(imageDoublePixels, index * LENGHT, imageData, 0, LENGHT);
        return imageData;
    }

    public static double[] getImageLabel(int index) {
        for (int i = 0; i < 10; i++) {
            LABEL_ARRAY[i] = label[index] == i ? 1 : 0;
        }
        return LABEL_ARRAY;
    }

    private static void loadImageData() {
        String workingDir = new File("").getAbsolutePath();
        File imagesFile = new File(workingDir + "/res/train-images.idx3-ubyte");
        int num = (int) (imagesFile.length() - 16);
        imageBytePixels = new byte[num];
        imageDoublePixels = new double[num];
        try {
            FileInputStream fileInputStream = new FileInputStream(imagesFile);
            fileInputStream.skip(16);
            int result;
            int offset = 0;
            while (-1 != (result = fileInputStream.read(imageBytePixels, offset, num - offset))) {
                offset += result;
                if (num == offset) break;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (int i = 0; i < num; i++) {
            imageDoublePixels[i] = imageBytePixels[i] / 128d;
        }
    }

    private static void loadLabelData() {
        String workingDir = new File("").getAbsolutePath();
        File imagesFile = new File(workingDir + "/res/train-labels.idx1-ubyte");
        int num = (int) (imagesFile.length() - 8);
        label = new byte[num];
        try {
            FileInputStream fileInputStream = new FileInputStream(imagesFile);
            fileInputStream.skip(8);
            int result;
            int offset = 0;
            while (-1 != (result = fileInputStream.read(label, offset, num - offset))) {
                offset += result;
                if (num == offset) break;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        loadData();
        test();
    }

    //控制台输出图片&标签
    private static void test() {
        int i = 0;
        while (true) {
            printImage(i);
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            i++;
        }
    }

    public static void printImage(int index) {
        System.out.println("label:" + Arrays.toString(getImageLabel(index)));
        double[] bytes = getImageData(index);
        assert bytes.length == 784;
        printImage(bytes);
    }

    public static void printImage(double[] bytes) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                if (bytes[i * 28 + j] >= 0) {
                    stringBuilder.append("■■");
                } else {
                    stringBuilder.append("  ");
                }
            }
            stringBuilder.append('\n');
        }
        System.out.println(stringBuilder.toString());
    }
}
