package priv.wjf.project.SparkNewsEventDetection;

import java.util.Calendar;

public class TimeConversion 
{
	/**
	 * 获取字符串日期的毫秒数
	 * @param time
	 * @return
	 */
	public static long getMilliseconds(String time) 
	{
		//201711010001
		int year = Integer.parseInt(time.substring(0, 4));
		int month = Integer.parseInt(time.substring(4, 6));
		int date = Integer.parseInt(time.substring(6, 8));
		int hourOfDay = Integer.parseInt(time.substring(8, 10));
		int minute = Integer.parseInt(time.substring(10, 12));
		Calendar calendar = Calendar.getInstance();
		calendar.set(year, month-1, date, hourOfDay, minute);
		
		return calendar.getTimeInMillis();
	}
}
