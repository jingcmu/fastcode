package mapred.ngramcount;

import java.io.IOException;

import mapred.util.Tokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class NgramCountMapper extends Mapper<LongWritable, Text, Text, NullWritable> {

	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] words = Tokenizer.tokenize(line);

		// get the number N from the Configuration
		Configuration conf = context.getConfiguration();
		final int N = Integer.parseInt(conf.get("N"));
		for (int i = 0; i <= words.length - N; i++){
			StringBuilder nGramSequence = new StringBuilder();
			for (int j = 0; j < N; j++){
				nGramSequence.append(words[i+j]);
			}
			context.write(new Text(nGramSequence.toString()), NullWritable.get());
		}
		/*
		for (String word : words)
			context.write(new Text(word), NullWritable.get());
		*/
	}
}
