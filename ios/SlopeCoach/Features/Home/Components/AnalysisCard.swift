import SwiftUI

struct AnalysisCard: View {
    let analysis: AnalysisSummary

    var body: some View {
        HStack(spacing: 14) {
            Image(analysis.thumbnailName)
                .resizable()
                .scaledToFill()
                .frame(width: 92, height: 92)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(alignment: .bottomTrailing) {
                    Image(systemName: "play.fill")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.white)
                        .padding(7)
                        .background(.ultraThinMaterial, in: Circle())
                        .padding(6)
                }

            VStack(alignment: .leading, spacing: 7) {
                Text(analysis.title)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Color.slopeNavy)
                    .lineLimit(1)

                Text(analysis.timestamp)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)

                HStack(spacing: 9) {
                    StatusBadge(status: analysis.status)

                    statusDetail
                }
            }

            Spacer(minLength: 0)
        }
        .padding(12)
        .background(.white, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.slopeBorder, lineWidth: 1)
        }
        .shadow(color: Color.slopeNavy.opacity(0.045), radius: 12, y: 5)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var statusDetail: some View {
        switch analysis.status {
        case .ready(let score):
            Text("\(score) Score")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Color.slopeNavy)
        case .partial(let reason):
            Text(reason)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.slopeWarning)
                .lineLimit(1)
        }
    }
}
