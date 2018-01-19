package com.ilioili;

import java.util.List;
import java.util.Random;

/**
 * Created by ilioili on 2018/1/3.
 */
public class MathUtil {

    public static double sigmod(double x) {
        return 1.0d / (1 + Math.exp(-x));
    }

    public static double sigmodPrime(double x) {
        return sigmod(x) * (1 - sigmod(x));
    }

    private static Random random = new Random(System.currentTimeMillis());

    public static void random(double[] array) {

        for (int i = 0; i < array.length; i++) {
            array[i] = random.nextDouble();
        }
    }

    public static double[] softmax(double[] data) {
        double[] result = new double[data.length];
        double sum = 0;
        for (int i = 0; i < result.length; i++) {
            result[i] = Math.exp(data[i]);
            sum += result[i];
        }
        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }
        return result;
    }

    public static int max(double[] data) {
        int index = 0;
        double max = data[0];
        for (int i = 1; i < data.length; i++) {
            if (max < data[i]) {
                max = data[i];
                index = i;
            }
        }
        return index;
    }
}
