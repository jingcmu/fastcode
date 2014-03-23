package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class SimilarityMapper extends
		Mapper<LongWritable, Text, IntWritable, Text> {

	// Map<String, Integer> jobFeatures = null;
	String tag = null;
	Map<String, Integer> tagFeatures = null;

	/**
	 * We compute the inner product of feature vector of every hashtag with that
	 * of the specific tag
	 */
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] hashtag_featureVector = line.split("\\s+", 4);
		String hashtag1 = hashtag_featureVector[0];

		String hashtag2 = hashtag_featureVector[2];
		
		// ignore self comparison
		if (hashtag1.equals(hashtag2)) {
			return;
		}
		
		Map<String, Integer> features1 = parseFeatureVector(hashtag_featureVector[1]);

		Map<String, Integer> features2 = parseFeatureVector(hashtag_featureVector[3]);

		Integer similarity = computeInnerProduct(features1, features2);
		
		// ignore 0 similarity
		if (similarity==0) {
			return;
		}
		
		context.write(new IntWritable(similarity), new Text(hashtag1 + "\t"
				+ hashtag2));
	}

	/**
	 * This function is ran before the mapper actually starts processing the
	 * records, so we can use it to setup the tag feature vector.
	 * 
	 * Loads the feature vector for hashtag this tag into mapper's memory
	 */
	@Override
	// protected void setup(Context context) {
	// String jobFeatureVector = context.getConfiguration().get(
	// "jobFeatureVector");
	// jobFeatures = parseFeatureVector(jobFeatureVector);
	// }
	protected void setup(Context context) {
//		String tagFeatureVector = context.getConfiguration().get(
//				"tagFeatureVector");
//		tag = context.getConfiguration().get("tag");
//		tagFeatures = parseFeatureVector(tagFeatureVector);
	}

	/**
	 * De-serialize the feature vector into a map
	 * 
	 * @param featureVector
	 *            The format is "word1:count1;word2:count2;...;wordN:countN;"
	 * @return A HashMap, with key being each word and value being the count.
	 */
	private Map<String, Integer> parseFeatureVector(String featureVector) {
		Map<String, Integer> featureMap = new HashMap<String, Integer>();
		String[] features = featureVector.split(";");
		for (String feature : features) {
			String[] word_count = feature.split(":");
			featureMap.put(word_count[0], Integer.parseInt(word_count[1]));
		}
		return featureMap;
	}

	/**
	 * Computes the dot product of two feature vectors
	 * 
	 * @param featureVector1
	 * @param featureVector2
	 * @return
	 */
	private Integer computeInnerProduct(Map<String, Integer> featureVector1,
			Map<String, Integer> featureVector2) {
		Integer sum = 0;
		for (String word : featureVector1.keySet())
			if (featureVector2.containsKey(word))
				sum += featureVector1.get(word) * featureVector2.get(word);

		return sum;
	}
}
