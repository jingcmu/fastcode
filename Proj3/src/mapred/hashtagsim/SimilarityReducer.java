package mapred.hashtagsim;

import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.IntWritable;

public class SimilarityReducer extends Reducer<Text, IntWritable, IntWritable, Text> {

	@Override
	protected void reduce(Text key, Iterable<IntWritable> values, Context context)
			throws IOException, InterruptedException {		
		int newValue = 0;
		for (IntWritable value : values) {
			newValue += value.get();
		}
		
		context.write(new IntWritable(newValue), key);
	}
}
