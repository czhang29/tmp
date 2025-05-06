from flask import Flask, render_template, session, redirect, url_for, request, jsonify

app = Flask(__name__)

app.secret_key = 'replace-this-with-a-real-secret-key'

# Define lessons (Include all lessons 1 through 6)
lessons = {
    "1": {
        "title": "Introduction",
        "content": "This guide will help you achieve ideal meditation posture using real-time feedback.",
        "image": "images/intro_image.jpg"
    },
    "2": {
        "title": "Benefits & Disadvantages",
        "content": "Benefits of good posture and consequences of poor posture.",
        "image": "images/benefits_image.jpg"
    },
    "3": {
        "title": "Body Position Overview",
        "content": "Proper alignment creates a stable foundation for your practice.",
        # "image": "/static/images/body_map.png" # Silhouette is part of the template now
    },
    "4": {
        "title": "Head & Neck Position",
        "content": """
          <ul class="list-unstyled">
              <li><strong>Chin Tuck:</strong> Gently tuck your chin towards your throat, lengthening the back of your neck. Imagine a string pulling the crown of your head upwards.</li>
              <li><strong>Level Gaze:</strong> Keep your gaze soft and level, not looking up or down excessively.</li>
              <li><strong>Relaxed Jaw:</strong> Ensure your jaw is unclenched and your facial muscles are relaxed.</li>
          </ul>
        """,
        "image": "images/head_position_image.jpg"
    },
    "5": {
        "title": "Shoulders & Arms Position",
        "content": """
          <ul class="list-unstyled">
              <li><strong>Shoulders Down & Back:</strong> Gently roll your shoulders up, back, and then let them relax down, away from your ears. Avoid hunching.</li>
              <li><strong>Open Chest:</strong> Allow your chest to be open but relaxed, not puffed out rigidly.</li>
              <li><strong>Arm Placement:</strong> Rest your hands comfortably in your lap or on your knees, palms up or down, whichever feels more natural. Keep elbows slightly bent and relaxed.</li>
          </ul>
        """,
        "image": "images/shoulders_position_image.jpg"
    },
    "6": {
        "title": "Spine",
        "content": """
          <ul class="list-unstyled">
              <li><strong>Natural Curve:</strong> Maintain the natural curves of your spine. Your lower back should have a slight inward curve, while your upper back has a slight outward curve.</li>
              <li><strong>Upright Posture:</strong> Sit or stand tall, as if a string were pulling the crown of your head toward the ceiling. This helps lengthen your spine and prevent slouching.</li>
              <li><strong>Core Engagement:</strong> Gently engage your core muscles to support your spine. This doesn't mean tensing rigidly, but maintaining a natural support for your back.</li>
          </ul>
        """,
        "image": "images/spine_position.jpg"
    }
}

# Define total steps for progress calculation
TOTAL_LEARNING_STEPS = 7 # 6 lessons + 1 practice page

@app.route('/')
def homepage():
    # Clear the learning flow flag when user returns to homepage
    session.pop('in_learning_flow', None) # Remove the key if it exists, do nothing otherwise
    progress_percent = 0
    return render_template('homepage.html', progress_percent=progress_percent)

@app.route('/learn/<lesson_number>')
def learn(lesson_number):
    lesson_data = lessons.get(lesson_number)
    if not lesson_data:
        # If lesson not found, clear flag and show 0 progress
        session.pop('in_learning_flow', None)
        return render_template('homepage.html', progress_percent=0), 404

    progress_percent = 0 # Default

    # Set flag at the start of learning
    if lesson_number == '1':
        session['in_learning_flow'] = True

    # Check if user is in the learning flow
    if session.get('in_learning_flow'):
        try:
            lesson_num_int = int(lesson_number)
            progress_percent = min(round((lesson_num_int / TOTAL_LEARNING_STEPS) * 100), 100)
        except ValueError:
            progress_percent = 0

    return render_template('learn.html',
                           lesson=lesson_data,
                           lesson_number=str(lesson_number),
                           lessons=lessons,
                           progress_percent=progress_percent)

@app.route('/spine')
def spine():
    progress_percent = min(round((6 / TOTAL_LEARNING_STEPS) * 100), 100) if session.get('in_learning_flow') else 0
    return render_template('spine.html', progress_percent=progress_percent)

@app.route('/practice')
def practice():
    progress_percent = 0 # Default

    # Show 100% progress ONLY if user was in the learning flow
    if session.get('in_learning_flow'):
        progress_percent = 100
    
    # Clear any previous practice focus
    session['practice_focus'] = 'all'
    
    return render_template('practice.html', progress_percent=progress_percent)

@app.route('/practice/head_neck')
def practice_head_neck():
    progress_percent = 0
    if session.get('in_learning_flow'):
        progress_percent = min(round((4 / TOTAL_LEARNING_STEPS) * 100), 100)
    
    session['practice_focus'] = 'head_neck'
    return render_template('practice_focused.html', 
                           focus="Head & Neck", 
                           title="Head & Neck Practice", 
                           progress_percent=progress_percent)

@app.route('/practice/shoulders')
def practice_shoulders():
    progress_percent = 0
    if session.get('in_learning_flow'):
        progress_percent = min(round((5 / TOTAL_LEARNING_STEPS) * 100), 100)
    
    session['practice_focus'] = 'shoulders'
    return render_template('practice_focused.html', 
                           focus="Shoulders", 
                           title="Shoulders Practice", 
                           progress_percent=progress_percent)

@app.route('/practice/spine')
def practice_spine():
    progress_percent = 0
    if session.get('in_learning_flow'):
        progress_percent = min(round((6 / TOTAL_LEARNING_STEPS) * 100), 100)
    
    session['practice_focus'] = 'spine'
    return render_template('practice_focused.html', 
                           focus="Spine", 
                           title="Spine Practice", 
                           progress_percent=progress_percent)

@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    """API endpoint to save practice feedback data"""
    if request.method == 'POST':
        try:
            # Get feedback data from request
            feedback_data = request.get_json()
            
            # Store in session
            for key, value in feedback_data.items():
                session[key] = value
                
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@app.route('/feedback_summary')
def feedback_summary():
    """Show practice feedback summary"""
    # Default values if session doesn't have data
    if 'head_neck_score' not in session:
        # For demo purposes, set some sample scores
        session['head_neck_score'] = 75
        session['shoulders_score'] = 60
        session['spine_score'] = 80
        session['overall_score'] = 72
        session['practice_duration'] = '3:45'
        
        # Sample feedback
        session['head_neck_feedback'] = "Your head and neck position is good, but could use some improvement. Try to keep your chin slightly tucked and ears level."
        session['shoulders_feedback'] = "Your shoulders show some tension. Practice relaxing them down and back, keeping them level with each other."
        session['spine_feedback'] = "Your spine maintains good alignment. Continue sitting tall with good core engagement."
        session['overall_feedback'] = "Your overall posture is good with some areas that could use improvement. With regular practice, you'll develop better postural habits."
    
    return render_template('feedback_summary.html')

@app.route('/results')
def results():
    """Show detailed results page"""
    # Ensure we have basic data for results
    if 'head_neck_score' not in session:
        # If no scores are set, redirect to summary to initialize defaults
        return redirect(url_for('feedback_summary'))
    
    # For demo, add more detailed metrics if not present
    if 'head_alignment' not in session:
        # Head & Neck details
        session['head_alignment'] = 'Good'
        session['head_alignment_score'] = 70
        session['neck_tension'] = 'Slightly Tense'
        session['neck_tension_score'] = 65
        session['gaze_direction'] = 'Level'
        session['gaze_direction_score'] = 80
        
        # Shoulders details
        session['shoulder_level'] = 'Slightly Uneven'
        session['shoulder_level_score'] = 60
        session['shoulder_tension'] = 'Tense'
        session['shoulder_tension_score'] = 50
        session['chest_openness'] = 'Partially Open'
        session['chest_openness_score'] = 65
        
        # Spine details
        session['spine_curvature'] = 'Natural'
        session['spine_curvature_score'] = 85
        session['lower_back_support'] = 'Moderate Support'
        session['lower_back_support_score'] = 70
        session['stability'] = 'Stable'
        session['stability_score'] = 80
        
        # Recommendations
        session['head_neck_recommendation'] = 'Practice daily chin tucks to strengthen your neck and improve alignment.'
        session['shoulders_recommendation'] = 'Perform shoulder rolls and gentle stretches throughout the day to release tension.'
        session['spine_recommendation'] = 'Strengthen your core with regular exercises to improve spine stability.'
        
        # Overall plan
        session['recommendation_1'] = 'Practice 5 minutes of posture awareness at your desk'
        session['recommendation_2'] = 'Do 2-3 gentle neck stretches every hour'
        session['recommendation_3'] = 'Strengthen your core with 10 minutes of exercises daily'
    
    return render_template('results.html')

# Add a CSS color scheme route for UI customization
@app.route('/set_theme', methods=['POST'])
def set_theme():
    """Set color theme preference"""
    if request.method == 'POST':
        theme = request.form.get('theme', 'default')
        session['color_theme'] = theme
        return redirect(request.referrer or url_for('homepage'))
    
    return redirect(url_for('homepage'))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)