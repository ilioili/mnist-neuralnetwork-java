package com.ilioili;

import java.text.DecimalFormat;

/**
 * Created by ilioili on 2018/1/3.
 */
public class ArrayPrintUtil {
    public static String toString(double[] array, String format) {
        DecimalFormat decimalFormat = new DecimalFormat(format);
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append('[');
        for (double d : array) {
            stringBuilder.append(decimalFormat.format(d)).append(", ");
        }
        stringBuilder.setLength(stringBuilder.length()-1);
        stringBuilder.append(']');
        return stringBuilder.toString();
    }
}
