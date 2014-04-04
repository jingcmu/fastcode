package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class SimilarityMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	Map<String, Integer> jobFeatures = null;

	/**
	 * We compute the inner product of feature vector of every hashtag with that
	 * of #job
	 */
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] hashtag_featureVector = line.split("\\s+", 2);
		ArrayList<String> tags = new ArrayList<String>();
		ArrayList<Integer> values = new ArrayList<Integer>();
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
		context.write(new Text(newTag.toString()), new IntWritable(similarity));						
			}
		}
	}

	/**
	 * This function is ran before the mapper actually starts processing the
	 * records, so we can use it to setup the job feature vector.
	 * 
	 * Loads the feature vector for hashtag #job into mapper's memory
	 */
	@Override
	protected void setup(Context context) {
			
	}

	/**
	 * De-serialize the feature vector into a map
	 * 
	 * @param featureVector
	 *            The format is "word1:count1;word2:count2;...;wordN:countN;"
	 * @return A HashMap, with key being each word and value being the count.
	 */
	private Map<String, Integer> parseFeatureVector(String featureVector, ArrayList<String> keys) {
		ArrayList<String> tags = new ArrayList<String>();
		ArrayList<Integer> values = new ArrayList<Integer>();
		Map<String, Integer> featureMap = new HashMap<String, Integer>();
		String[] hashTags = featureVector.split(";");
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
				keys.add(newTag.toString());
				featureMap.put(newTag.toString(), similarity);						
			}
		}
		return featureMap;
	}

	/**
	 * Computes the dot product of two feature vectors
	 * @param featureVector1
	 * @param featureVector2
	 * @return 
	 */
	/*
	private Integer computeInnerProduct(Map<String, Integer> featureVector1,
			Map<String, Integer> featureVector2) {
		Integer sum = 0;
		for (String word : featureVector1.keySet()) 
			if (featureVector2.containsKey(word))
				sum += featureVector1.get(word) * featureVector2.get(word);
		
		return sum;
	}*/
}

