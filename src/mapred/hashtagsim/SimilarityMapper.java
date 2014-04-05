package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.*;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class SimilarityMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	Map<String, Integer> jobFeatures = null;
	private Pattern pattern = Pattern.compile("\\s+");

	/**
	 * We compute the inner product of feature vector of every hashtag with that
	 * of #job
	 */
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] hashtag_featureVector = pattern.split(line, 2);
		ArrayList<String> tags = new ArrayList<String>();
		ArrayList<Integer> values = new ArrayList<Integer>();
		Text keys = new Text();
		IntWritable valueOut = new IntWritable();
		String[] hashTags = hashtag_featureVector[1].split(";");
		Arrays.sort(hashTags);
		for (String hashTag : hashTags) {
			String[] tag_count = hashTag.split(":");
			tags.add(tag_count[0]);
			values.add(Integer.parseInt(tag_count[1]));
		}
		StringBuffer newTag = new StringBuffer();
		for(int i=0; i<tags.size()-1; i++) {
			for(int j=i+1; j<tags.size(); j++) {
				int similarity = values.get(i)*values.get(j);
				newTag.setLength(0);
				newTag.append(tags.get(i));
				newTag.append(" ");
				newTag.append(tags.get(j));
				keys.set(newTag.toString());
				valueOut.set(similarity);
				context.write(keys, valueOut);						
			}
		}
	}

}

